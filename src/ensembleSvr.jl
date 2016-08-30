using DataFrames
using HypothesisTests
using Lazy
using Logging
using MLBase
using Optim
using PValueAdjust

@everywhere include("DataInfo.jl")
@everywhere include("helpers.jl")
@everywhere include("svrPipe.jl")


@everywhere function findOptimumC!(pipe::Pipeline, cvg_factory::Function;
                      eval_scores::Function = arr -> OneSampleTTest(arr).t,
                      c_grid::AbstractVector = [logspace(-6, 2, 9); logspace(-6, 2, 9).*5]
                      )
  n_samples = 30
  train_ratio = .8
  cvg = cvg_factory(n_samples, train_ratio, pipe.truths)

  state_combos = stateCombos(:C => c_grid)
  test_evals = state_combos |> length |> zeros
  updateTestEvals! = (_, test_scores, a, ix) -> test_evals[ix] = eval_scores(test_scores)

  _, _, combos = evalModel(pipe, cvg, state_combos, on_combo_complete=updateTestEvals!)

  combos_sorted = combos[sortperm(test_evals)]

  C =  combos_sorted[end][:C]

  info(@sprintf "best C: %3.2e, with t: %3.2e" C maximum(test_evals) )

  C
end


typealias Inds Vector{Int64}
function runPipeline(X::Matrix,
                  y::Vector,
                  pipe_factory::Function,
                  get_C::Function,
                  train_ixs_array;
                  y_perm_ixs::Matrix = repmat(1:length(y), 1, n_samples),
                  n_samples::Int64=1000)

  coefficients = SharedArray(Float64, size(X, 2), n_samples)
  cs = SharedArray(Float64, n_samples)

  prev_show = 0
  all_ixs = 1:length(y)
  scores::Vector{Float64} = pmap(1:n_samples) do s::Int64
    pipe = pipe_factory(X, y, s)

    function fit!(x_train_ixs, y_train_ixs)
      X_fit, y_fit = X[x_train_ixs, :], y[y_train_ixs]
      c = get_C(X_fit, y_fit)
      @show c
      cs[s] = c

      paramState!(pipe, :C => cs[s])
      pipeFit!(pipe, x_train_ixs, y_train_ixs)
    end

    if (s - prev_show)/n_samples > .1
      prev_show = s
      println("at $s out of $(n_samples)")
    end

    train_ixs = train_ixs_array[s]
    test_ixs = setdiff(all_ixs, train_ixs)

    fit!(train_ixs, y_perm_ixs[train_ixs, s])

    coefs = paramState(pipe, :classifier)[:coef_]
    variant_features = paramState(pipe, :variant_features)
    coefficients[variant_features, s] = coefs

    pipeTest(pipe, test_ixs, y_perm_ixs[test_ixs, s])
  end

  scores, coefficients, cs
end


function calcEnsemble(preds_lesion::AbstractMatrix, preds_conn::AbstractMatrix, truths::Vector;
  weights::Dict{DataSet, Float64} = Dict(lesion=>.5, conn=>.5))

  @assert size(preds_conn) == size(preds_lesion)

  n_samples = size(preds_lesion, 2)

  preds = repmat([NaN], size(preds_conn)...)
  r2s = zeros(Float64, n_samples)

  preds_binary = repmat([NaN], size(preds_conn)...)
  accuracies = zeros(Float64, n_samples)

  for s in 1:n_samples
    lesion_ixs = [!isnan(i) for i in preds_lesion[:, s]]
    conn_ixs = [!isnan(i) for i in preds_conn[:, s]]

    @assert lesion_ixs == conn_ixs

    preds[conn_ixs, s] = preds_lesion[lesion_ixs, s] .* weights[lesion] +
      preds_conn[conn_ixs, s] .* weights[conn]

    preds_binary[conn_ixs, s] = preds[conn_ixs, s] .> 0

    r2s[s] = r2score(truths[conn_ixs], preds[conn_ixs, s])
    accuracies[s] = accuracy(truths[conn_ixs] .> 0,
      [i  > 0 for i  in preds_binary[conn_ixs, s]])
  end

  r2s, preds, accuracies, preds_binary
end


@everywhere looFactory(n_samples, train_ratio, ys) = @> ys length LOOCV
@everywhere stratifiedRandomSubFactory(n_samples, train_ratio, ys) = StratifiedRandomSub(
  ys .> 0, round(Int64, length(ys) * train_ratio), n_samples)


function runPerms(n_perms; sample_size=50)
  y_perm_ixs = zeros(Int64, sample_size, n_perms)
  for c in 1:n_perms
    y_perm_ixs[:, c] = randperm(sample_size)
  end
  runClass(n_perms, y_perm_ixs=Nullable(y_perm_ixs))
end


function runClass(n_samples::Int64;
  region::Region=left_select2,
  y_col::Symbol=:adw_diff_wps,
  cvg_factory=stratifiedRandomSubFactory,
  train_ixs_array::Nullable{Vector{Inds}} = Nullable{Vector{Inds}}(),
  y_perm_ixs::Nullable{Matrix{Int64}}=Nullable{Matrix{Int64}}())

  conn_di, lesion_di = map([conn, lesion]) do ds::DataSet
    DataInfo(adw, diff_wps, all_subjects, region, ds)
  end

  common_ids = getCommonIds([conn_di, lesion_di])
  x_conn, x_lesion, y = begin
    conn_full = getFull(conn_di)
    lesion_full = getFull(lesion_di)

    x_conn = conn_full[getPredictors(conn_di)] |> Matrix
    x_lesion = lesion_full[getPredictors(lesion_di)] |> Matrix

    @assert conn_full[y_col] == lesion_full[y_col]

    x_conn, x_lesion, conn_full[y_col] |> Vector
  end

  x_map = Dict{DataSet, Matrix}(conn=>x_conn, lesion=>x_lesion)

  sample_size = length(y)

  function catPipeline(X, y, on_continuous_pred_calc=(arr, ixs, p)->())
    pipe = svrPipeline(X, y)
    trainContinuousTestCategorical(pipe,
      state_keys=[:C, :classifier, :variant_features],
      on_continuous_pred_calc=on_continuous_pred_calc
      )
  end

  initPreds() = @>> n_samples repmat(Float64[NaN], sample_size) convert(SharedArray)

  continuous_preds_map = Dict{DataSet, SharedArray}(conn => initPreds(),
    lesion=>initPreds())

  train_ixs_arr = get(train_ixs_array, cvg_factory(n_samples, .8, y) |> collect)

  function preds(ds::DataSet, find_c_cvg_factory)
    get_c(X_fit, y_fit) = findOptimumC!(catPipeline(X_fit, y_fit),
      find_c_cvg_factory)

    updatePredsMapGen!(s) = (preds, ixs, p) -> continuous_preds_map[ds][ixs, s] = preds

    x::Matrix = x_map[ds]

    runPipeline(x,
      y,
      (x, y, s) -> catPipeline(x, y, updatePredsMapGen!(s)),
      get_c,
      train_ixs_arr,
      n_samples=n_samples,
      y_perm_ixs=get(y_perm_ixs, repmat(1:length(y), 1, n_samples))
      )
  end

  println("predicting lesion")
  accuracies_lesion, coefs_lesion, cs_lesion = preds(lesion, stratifiedRandomSubFactory)

  println("predicting connectivity")
  accuracies_conn, coefs_conn, cs_conn = preds(conn, looFactory)

  _, preds_55, accuracies_55, _ = calcEnsemble(continuous_preds_map[lesion],
    continuous_preds_map[conn], y)

  ret = Dict()
  ret[:preds_continuous] = Dict(:conn => continuous_preds_map[conn],
    :lesion => continuous_preds_map[lesion],
    :ens => preds_55
    )

  ret[:accuracies] = Dict(:conn => accuracies_conn,
    :lesion => accuracies_lesion,
    :ens => accuracies_55)

  ret[:cs] = Dict(:conn => cs_conn, :lesion => cs_lesion)
  ret[:coefs] = Dict(:conn => coefs_conn, :lesion => coefs_lesion)

  ret[:ids] = common_ids

  ret[:train_ixs_array] = train_ixs_arr

  ret
end


function calcCoefStats(coefs_df::DataFrame)
  coefs_df_long = @> coefs_df melt rename!(:variable, :predictor)

  groupByPred(fn, s) = @> coefs_df_long by(:predictor, df -> fn(df[:value])) rename!(:x1, s)

  n_samples = size(coefs_df, 1)

  mean_df = groupByPred(mean, :mean)
  standard_dev_df = groupByPred(std, :standard_dev)

  pos_ratio = groupByPred(vs -> (sum(vs .> 0)+1)/(n_samples+1), :pos_ratio)
  neg_ratio = groupByPred(vs -> (sum(vs .< 0)+1)/(n_samples+1), :neg_ratio)

  ret = @>> mean_df begin
    join(standard_dev_df, on=:predictor)
    join(pos_ratio, on=:predictor)
    join(neg_ratio, on=:predictor)
  end

  ret[:pos_ratio_adj] = padjust(ret[:pos_ratio], BenjaminiHochberg)
  ret[:neg_ratio_adj] = padjust(ret[:neg_ratio], BenjaminiHochberg)

  ret
end


dropnan(arr) = arr[Bool[!isnan(i) for i in arr]]


predAgg(agg_fn, preds_matrix) = map(1:size(preds_matrix, 1)) do r
  preds_matrix[r, :] |> dropnan |> agg_fn
end

predAvg(preds_matrix) = predAgg(mean, preds_matrix)

predStdErrs(preds_matrix) = predAgg(preds_matrix) do v
  std(v)/(v |> length |> sqrt)
end

function coefsTable(run_class_ret, k::DataSet, target=diff_wps)
  predictors = @>> k DataInfo(adw, target, all_subjects, left_select2) getPredictors
  v = run_class_ret[:coefs][Symbol(k)]
  @> v' DataFrame d -> rename!(d, names(d), predictors)
end


function coefsTable(run_class_ret, k::Symbol, target=diff_wps)
  ds::DataSet = k == :conn ? conn : lesion
  coefsTable(run_class_ret, ds, target)
end


function saveRunClass(run_class_ret::Dict, dir_name::ASCIIString)

  dest_dir = begin
    par_dir = @> data_dir() joinpath("step4", "svr_classify")
    joinpath(par_dir, dir_name)
  end

  mkdir(dest_dir)

  destF(f_name) = joinpath(dest_dir, f_name)

  for (k, v) in run_class_ret[:accuracies]
    @> "accuracies_$(k).csv" destF writecsv(v)
  end

  for (k, cs) in run_class_ret[:cs]
    @> "cs_$(k).csv" destF writecsv(cs)
  end

  for (k, v) in run_class_ret[:preds_continuous]
    @> "predictions_continuous_$(k).csv" destF writecsv(v)
  end

  for (k::Symbol, v) in run_class_ret[:coefs]
    df = coefsTable(run_class_ret, k)
    @> "predictors_$(k).csv" destF writetable(df)
    @> "predictors_stats_$(k).csv" destF writetable(calcCoefStats(df))
  end

  @> "ids.csv" destF writecsv(run_class_ret[:ids])

end
