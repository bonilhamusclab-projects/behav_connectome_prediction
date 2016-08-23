using DataFrames
using HypothesisTests
using Lazy
using Logging
using MLBase
using Optim
using PValueAdjust

include("DataInfo.jl")
include("helpers.jl")
include("optimizeHelpers.jl")
include("svrClassify.jl")
include("svrBase.jl")


function calcScoresGen(pipe::Pipeline,
  cvg::CrossValGenerator,
  call_test::Bool=true)

  n_samples = length(cvg)
  n_subjects = length(pipe.truths)

  test(_, ixs::Vector{Int64}) = call_test ? pipeTest(pipe, ixs) : -1.

  function calcScores(C::Float64)
    paramState!(pipe, :C => C)

    train_scores = zeros(Float64, n_samples)
    fit_call_count::Int64 = 0
    fit(ixs::IXs) = begin
      fit_call_count += 1
      pipeFit!(pipe, ixs)

      train_scores[fit_call_count] = pipeTest(pipe, ixs)
    end

    test_scores = cross_validate(fit, test, n_subjects, cvg)

    debug(@sprintf "C: %3.2e, test mean: %3.2e test T: %3.2e, train mean: %3.2e" C mean(test_scores) OneSampleTTest(test_scores).t mean(train_scores))

    test_scores, train_scores
  end
end


function findOptimumC!(pipe::Pipeline, cvg_factory::Function;
                      score_train_scores::Function = mean,
                      score_test_scores::Function = arr -> OneSampleTTest(arr).t,
                      c_grid::AbstractVector = [logspace(-6, 2, 9); logspace(-6, 2, 9).*5]
                      )
  function calcSampleScoresGen(n_samples, train_ratio)
    cvg = cvg_factory(n_samples, train_ratio, pipe.truths)
    calcScoresGen(pipe, cvg)
  end

  calc30Scores = calcSampleScoresGen(30, .8)
  testScore(C::Float64) = calc30Scores(C)[1] |> score_test_scores

  C::Float64 = @>> c_grid begin
    map(c -> (c, testScore(c)))
    sort(by=ct -> ct[2], rev=true)
    map(ct -> ct[1])
    first
  end

  info(@sprintf "best C: %3.2e, with t: %3.2e, for dataset %s" C -1 * testScore(C) d)

  C
end


typealias DataSetMap Dict{DataSet, Float64}

function runPipeline(X::Matrix,
                  y::Vector,
                  pipe::Pipeline,
                  get_C::Function,
                  get_sample_ixs::Function;
                  n_samples::Int64=1000)

  coefficients = zeros(Float64, size(X, 2), n_samples)

  function fit!(x_train_ixs, y_train_ixs, sample_ix)
    X_fit, y_fit = X[x_train_ixs, :], y[y_train_ixs]
    C = get_C(X_fit, y_fit, sample_ix)

    paramState!(pipe, :C => C)
    pipeFit!(pipe, x_train_ixs, y_train_ixs)
  end

  test(x_test_ixs, y_test_ixs) = pipeTest(pipe, x_test_ixs, y_test_ixs)

  prev_show = 0
  scores::Vector{Float64} = map(1:n_samples) do s::Int64
    if (s - prev_show)/n_samples > .1
      prev_show = s
      println("at $s out of $(n_samples)")
    end
    (x_train_ixs, y_train_ixs), (x_test_ixs, y_test_ixs) = get_sample_ixs(s)

    fit!(x_train_ixs, y_train_ixs, s)

    coefs = paramState(pipe, :classifier)[:coef_]
    variant_features = paramState(pipe, :variant_features)
    coefficients[variant_features, s] = coefs

    test(x_test_ixs, y_test_ixs)
  end

  scores, coefficients
end


type State
  cs::Vector{Float64}
  getrepixs::Function
end

State(n_samples::Int64) = State(
  zeros(Float64, n_samples),
  i::Int64 -> error("not set yet")
)


function findOptimumCandUpdateStateGen(state::State,
    pipe_factory::Function, cvg_factory::Function)

  n_samples = state.cs |> length

  updateState!(sample_ix::Int64, C::Float64) = state.cs[sample_ix] = C

  function findOptimumCandUpdateState(X::Matrix, y::Vector, sample_ix::Int64)
    pipe = pipe_factory(X, y)

    C::Float64 = findOptimumC!(pipe, cvg_factory)

    updateState!(sample_ix, C)
    C
  end
end


function calcEnsemble(preds_lesion::Matrix, preds_conn::Matrix, truths::Vector;
  weights::Dict{DataSet, Float64} = Dict(lesion=>.5, conn=>.5))

  @assert size(preds_conn) == size(preds_lesion)

  n_samples = size(preds_lesion, 2)

  preds = repmat([NaN], size(preds_conn)...)
  r2s = zeros(Float64, n_samples)

  for s in 1:n_samples
    lesion_ixs = [!isnan(i) for i in preds_lesion[:, s]]
    conn_ixs = [!isnan(i) for i in preds_conn[:, s]]

    @assert lesion_ixs == conn_ixs

    preds[conn_ixs, s] = preds_lesion[lesion_ixs, s] .* weights[lesion] +
      preds_conn[conn_ixs, s] .* weights[conn]

    r2s[s] = r2score(truths[conn_ixs], preds[conn_ixs, s])
  end

  r2s, preds
end


looFactory(n_samples, train_ratio, ys) = @> ys length LOOCV
stratifiedRandomSubFactory(n_samples, train_ratio, ys) = StratifiedRandomSub(
  ys .> 0, round(Int64, length(ys) * train_ratio), n_samples)


function runClass(n_samples::Int64;
  cvg_factory=looFactory,
  find_c_cvg_factory=cvg_factory,
  y_ixs::Nullable{Vector{Int64}}=Nullable{Vector{Int64}}())

  state_map::Dict{DataSet, State} = Dict{DataSet, State}(
    conn => State(n_samples), lesion => State(n_samples))

  conn_di, lesion_di = map([conn, lesion]) do ds::DataSet
    DataInfo(adw, diff_wps, all_subjects, left_select2, ds)
  end

  common_ids = getCommonIds([conn_di, lesion_di])
  x_conn, x_lesion, y = begin

    x_conn, y_conn = getXyMat(conn_di, common_ids)
    x_lesion, y_lesion = getXyMat(lesion_di, common_ids)

    @assert y_conn == y_lesion

    x_conn, x_lesion, y_conn[get(y_ixs, 1:end)]
  end

  x_map = Dict{DataSet, Matrix}(conn=>x_conn, lesion=>x_lesion)

  sample_size = length(y)

  get_sample_ixs = @> n_samples cvg_factory(.8, y) samplesGen(sample_size)

  function catPipeline(X, y, on_continuous_pred_calc=(arr, ixs, p)->())
    pipe = svrPipeline(X, y)
    trainContinuousTestCategorical(pipe,
      state_keys=[:C, :classifier, :variant_features],
      on_continuous_pred_calc=on_continuous_pred_calc
      )
  end

  initPreds() = repmat([NaN], sample_size, n_samples)

  continuous_preds_map = Dict{DataSet, Matrix}(conn => initPreds(),
    lesion=>initPreds())

  function preds(ds::DataSet)
    state = state_map[ds]
    get_C! = findOptimumCandUpdateStateGen(state, catPipeline, find_c_cvg_factory)

    updatePredsMap!(preds, ixs, p) = continuous_preds_map[ds][ixs, p] = preds

    x::Matrix = x_map[ds]

    runPipeline(x,
      y,
      catPipeline(x, y, updatePredsMap!),
      get_C!,
      get_sample_ixs,
      n_samples=n_samples)
  end

  println("predicting lesion")
  accuracies_lesion, coefs_lesion = preds(lesion)

  println("predicting connectivity")
  accuracies_conn, coefs_conn = preds(conn)

  ret = Dict()
  ret[:preds_continuous] = Dict(:conn => continuous_preds_map[conn],
    :lesion => continuous_preds_map[lesion])

  accuracies_55, preds_55 = begin
    _, preds_continous = calcEnsemble(continuous_preds_map[lesion],
      continuous_preds_map[conn], y)

    ret[:preds_continuous][:ens] = preds_continous

    preds_ret = repmat([NaN], size(preds_continous)...)

    accuracies_ret = map(1:n_samples) do s
      valid_ixs = Bool[!isnan(i) for i in preds_continous[:, s]]

      preds_binary = preds_continous[valid_ixs, s] .>= 0
      preds_ret[valid_ixs, s] = preds_binary

      accuracy(y[valid_ixs] .>= 0, preds_binary)
    end

    accuracies_ret, preds_ret
  end

  ret[:accuracies] = Dict(:ens => accuracies_55, :conn => accuracies_conn, :lesion => accuracies_lesion)
  ret[:state] = Dict(conn_di => state_map[conn], lesion_di => state_map[lesion])
  ret[:coefs] = Dict(:conn => coefs_conn, :lesion => coefs_lesion)

  ret[:ids] = common_ids
  ret[:get_sample_ixs] = get_sample_ixs

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


predAvg(preds_matrix) = map(1:size(preds_matrix, 1)) do r
  preds_matrix[r, :] |> dropnan |> mean
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


function saveRunClass(run_class_ret::Dict;
  target::Target=diff_wps,
  score_fn::Symbol=:accuracies,
  prefix::ASCIIString="")

  datasetToDataInfo(ds::DataSet) = DataInfo(
    adw, target, all_subjects, left_select2, ds)

  dest_dir = @> data_dir() joinpath("step4", "svr_classify")
  destF(f_name) = joinpath(dest_dir,
    isempty(prefix) ?  f_name : "$(prefix)_$(f_name)")

  for (k, v) in run_class_ret[:accuracies]
    @> "accuracies_$(k).csv" destF writecsv(v)
  end

  for (di::DataInfo, s::State) in run_class_ret[:state]
    @> "cs_$(di).csv" destF writecsv(s.cs)
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
