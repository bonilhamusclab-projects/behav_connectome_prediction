using Colors
using DataFrames
using HypothesisTests
using Lazy
using MLBase
using PValueAdjust
using PyCall

include("DataInfo.jl")
include("helpers.jl")

@pyimport sklearn.svm as svm
LinearSVR = svm.LinearSVR


macro l2_from_true(x)
  :(sum( (y_true - $x).^2))
end

function r2_score(y_true::Vector{Float64}, y_pred::Vector{Float64})

  numerator::Float64 = @l2_from_true y_pred
  denominator::Float64 = @l2_from_true mean(y_true)

  1 - numerator/denominator
end


function get_Xy_mat(o::Outcome,
                    target::Target;
                    dataset::DataSet=conn,
                    region::Region=full_brain,
                    subject_group::SubjectGroup=all_subjects)
  get_Xy_mat(DataInfo(o, target, subject_group, region, dataset))
end


function pred_diff(m::Outcome;dataset::DataSet=conn)
  svr = LinearSVR()

  X, y = get_Xy_mat(m, diff_wpm, dataset=dataset)

  fit(inds::Vector{Int64}) = svr[:fit](X[inds, :], y[inds])

  test(c::PyObject, inds::Vector{Int64}) =
    r2_score(y[inds], c[:predict](X[inds, :]))

  num_samples = length(y)

  cross_validate(fit,
                 test,
                 num_samples,
                 Kfold(num_samples, 5))
end


function num_samples(cv::CrossValGenerator)
  @switch isa(cv, _) begin
    Kfold; length(cv.permseq)
    StratifiedRandomSub; sum(map(length, cv.idxs))
    cv.n
  end
end


function ratios_to_counts(ratios::AbstractVector{Float64}, n::Int64)
  @assert all( (ratios .<= 1.) &  (ratios .>= 0.) )

    get_count(r::Float64) = round(Int, r*n)::Int64

    map(get_count, ratios)
end


function learning_curve(svr::PyObject,
                        lc_cvg_gen::Function,
                        o::Outcome,
                        dataset::DataSet,
                        region::Region,
                        subject_group::SubjectGroup,
                        train_ratios::AbstractVector{Float64};
                        score_fn::Function=r2_score)

  X, y = get_Xy_mat(o, diff_wpm, dataset=dataset, region=region, subject_group=subject_group)
  num_samples::Int64 = length(y)

  train_sizes::Vector{Int64} = ratios_to_counts(train_ratios, num_samples)
  train_sizes = train_sizes[train_sizes .> 2]

  ret::DataFrame = begin
    mk_zeros = () -> zeros(Float64, length(train_sizes))

    DataFrame(train_mn=mk_zeros(),
              train_std=mk_zeros(),
              test_mn=mk_zeros(),
              test_std=mk_zeros(),
              train_size=train_sizes)
  end

  test(c::PyObject, inds::Vector{Int64}) =
    score_fn(y[inds], c[:predict](X[inds, :]))

  for (ix, t) in enumerate(train_sizes)
    cvg::CrossValGenerator = lc_cvg_gen(t)

    train_scores = Float64[]

    fit(inds::Vector{Int64}) = begin
      svr[:fit](X[inds, :], y[inds])
      train_score = test(svr, inds)
      train_scores = [train_scores; train_score]
      svr
    end

    test_scores = cross_validate(fit, test, t, cvg)

    ret[ix, :train_mn] = mean(train_scores)
    ret[ix, :train_std] = std(train_scores)
    ret[ix, :test_mn] = mean(test_scores)
    ret[ix, :test_std] = std(test_scores)
  end

  ret
end


function get_cvg_gen(cvgT::Type, n_folds::Int64)
  @assert cvgT <: CrossValGenerator
  @switch cvgT begin
    Kfold; (n_samples::Int64) -> Kfold(n_samples, n_folds)
    RandomSub; (n_samples::Int64) -> RandomSub(n_samples, round(Int64, .8 * n_samples), n_folds)
  end
end

get_cvg(cvgT::Type, n_folds::Int64, n_samples::Int64) = get_cvg_gen(cvgT, n_folds)(n_samples)


function learning_curve(o::Outcome;
                        dataset::DataSet=conn,
                        region::Region=left_select,
                        C=1.0,
                        subject_group::SubjectGroup=all_subjects,
                        seed::Nullable{Int}=Nullable(1234),
                        train_ratios::AbstractVector{Float64}=1./6:1./6:1.)
  isnull(seed) || srand(get(seed))

  svr = LinearSVR(C=C)
  lc_cvg_gen = get_cvg_gen(RandomSub, 25)
  learning_curve(svr, lc_cvg_gen, o, dataset, region, subject_group, train_ratios)
end


function learning_curve(di::DataInfo, C::Float64; seed::Nullable{Int}=Nullable(1234),
                        train_ratios::AbstractVector{Float64}=1./6:1./6:1.)
  learning_curve(di.outcome, region=di.region, C=C, subject_group=di.subject_group, seed=seed,
                 train_ratios=train_ratios)
end


function calc_coefs(d::DataInfo,
                    C::Float64;
                    n_perms::Int64=1000,
                    seed::Nullable{Int}=Nullable(1234))

  isnull(seed) || srand(get(seed))

  X::Matrix{Float64}, y::Vector{Float64} = get_Xy_mat(d)

  num_samples::Int64 = length(y)

  svr = LinearSVR(C=C)
  cvg::CrossValGenerator = get_cvg(RandomSub, n_perms, num_samples)

  predictors = get_predictors(d)
  n_predictors = length(predictors)

  coefs::DataFrame = begin
    ret = DataFrame()
    for p in predictors
      ###Hack to keep it float64 while being NA
      ret[p] = repmat([-Inf], n_perms)
      ret[ret[p] .== -Inf, :] = NA
    end
    ret
  end

  fit, test = begin

    state = Dict(:fit_call => 0)

    fit_fn(inds::Vector{Int64}) = begin
      state[:fit_call] += 1
      println(state[:fit_call])
      svr[:fit](X[inds, :], y[inds])
      for (ix, c) in enumerate(svr[:coef_])
        coefs[state[:fit_call], predictors[ix]] = c
      end
      svr
    end

    test_fn(c::PyObject, inds::Vector{Int64}) =
      r2_score(y[inds], c[:predict](X[inds, :]))

    (fit_fn, test_fn)
  end

  test_scores::Vector{Float64} = cross_validate(fit, test, num_samples, cvg)

  typealias HtInfo Dict{Symbol, Float64}
  ht_info(arr::Vector{Float64}) = begin
    ret = Dict{Symbol, Float64}()
    ht = OneSampleTTest(arr)

    for tail in [:right, :left, :both]
      p_sym = symbol(tail, :_p)
      ret[p_sym] = pvalue(ht, tail=tail)
    end

    ret[:t] = ht.t
    ret
  end

  pv(arr::Vector{Float64}, tail::Symbol=:both) = pvalue(OneSampleTTest(arr), tail=tail)

  prediction_info::DataFrame = begin
    ht = ht_info(test_scores)
    pi = DataFrame(
      mean=mean(test_scores),
      std=std(test_scores),
      t=ht[:t],
      right_p=ht[:right_p],
      left_p=ht[:left_p],
      both_p=ht[:both_p],
      num_perms=n_perms)
  end

  predictor_info::DataFrame = begin
    pr = DataFrame()
    pr[symbol(d.dataset)] = predictors
    pr[symbol(d.dataset, :_name)] = create_predictor_names(predictors, d.dataset)


    pred_apply(fn::Function) = [fn(dropna(coefs[p])) for p in predictors]

    pr[:mean] = pred_apply(mean)
    pr[:std] = pred_apply(std)

    hts::Vector{HtInfo} = pred_apply(ht_info)
    for k::Symbol in keys(hts[1])
      pr[k] = Float64[i[k] for i in hts]

      is_p_measure::Bool = endswith(string(k), "_p")
      if is_p_measure
        pr[symbol(k, :_adj)]= padjust(pr[k], BenjaminiHochberg)
      end
    end

    sort(pr, cols=:t)
  end

  Dict(symbol(d, :_predictors) => predictor_info,
       symbol(d, :_predictions) => prediction_info)

end


function calc_all_coefs(cs::Dict{DataInfo, Float64})

  ret = Dict{DataInfo, Dict{Symbol, DataFrame}}()
  for (di::DataInfo, C::Float64) in cs
    println(di)
    ret[di] = calc_coefs(di, C)
  end

  ret

end


function calc_lesion_coefs()
  mk_di(o::Outcome, r::Region) = DataInfo(o, diff_wpm, all_subjects, r, lesion)
  regions::Vector{Region} = Region[left_select, left, full_brain]
  cs::Dict{DataInfo, Float64} = [mk_di(o, r) => 10. for r in regions,
                                 o in Outcome[adw, atw]]
  calc_all_coefs(cs)
end


function save_calc_coefs(calc_all_coefs_ret::Dict{DataInfo, Dict{Symbol, DataFrame}})
  for di::DataInfo in keys(calc_all_coefs_ret)
    dir = joinpath("$(data_dir())/step4/svr/")
    isdir(dir) || mkpath(dir)
    for (k::Symbol, data::DataFrame) in calc_all_coefs_ret[di]
      f_name = "$k.csv"
      println(f_name)
      writetable(joinpath(dir, f_name), data)
    end
  end
end


function lc_flip(di::DataInfo, C::Float64, flip_ratio::Float64)
  ret = DataFrame()
  for C_tmp = [C/flip_ratio, C, C*flip_ratio]

    suff(k::Symbol) = if(C_tmp != C)
      C_tmp > C ? symbol(k, "_gt") : symbol(k, "_lt") else
      k end

    lc = learning_curve(di, C_tmp, train_ratios=5./6:1./6:1.)[end, :]
    ret[suff(:C)] = C_tmp

    for n in names(lc)
      ret[suff(n)] = lc[n]
    end

  end

  for f in fieldnames(di)
    ret[f] = di[f]
  end

  ret
end


###Hypothesis: only brain region determines optimal C for a measure
function param_test()
  best_Cs = Dict(
    atw => Dict(
      left => 5e-4,
      left_select => 5e-3,
      full_brain => 5e-4
      ),
    adw => Dict(
      left => 5e-4,
      left_select => 5e-3,
      full_brain => 5e-4
      )
    )

  specials = Dict(
    DataInfo(adw, diff_wpm, improved, full_brain) => 1e-3
    )

  data_infos = for_all_combos(targets=[diff_wpm])

  ret = DataFrame()

  for di in data_infos
    println()
    println("#####")

    C = in(di, keys(specials)) ? specials[di] : best_Cs[di.outcome][di.region]

    println("di: $di, C: $C")
    lc::DataFrame = lc_flip(di, C, 5.)
    ret = isempty(ret) ? lc : vcat(ret, lc)

    println()
  end

  sort(ret, cols=fieldnames(DataInfo))
end


function plot_param_test(pt_res::DataFrame)
  function mk_layer_di(col::Symbol, clr::AbstractString)
    col_std::Symbol = replace("$col", "_mn", "_std")
    ymax = pt_res[col] + pt_res[col_std]
    ymin = pt_res[col] - pt_res[col_std]
    di_strings = [to_string(DataInfo(m[2], diff_wpm, s[2], r[2]))[1:end-9]
                  for (m, s, r) in eachrow(pt_res[[:outcome, :subject_group, :region]])]

    layer(x, x=di_strings, y=col, ymax=ymax, ymin=ymin,
          Geom.line, Geom.point,
          Theme(default_color=parse(Colorant, clr))
          )
  end

  plot_test() = plot(mk_layer_di(:test_mn, "#00FF00"),
                     mk_layer_di(:test_mn_lt, "#FF0000"),
                     mk_layer_di(:test_mn_gt, "#0000FF"))

  plot_train() = plot(mk_layer_di(:train_mn, "#00FF00"),
                      mk_layer_di(:train_mn_lt, "#FF0000"),
                      mk_layer_di(:train_mn_gt, "#0000FF"))

  (plot_test, plot_train)

end


function ensemble(outcome::Outcome, region::Region=left_select,
                  conn_C::Float64=5e-3, lesion_C::Float64=50.;
                  weights::Dict{DataSet, Float64} = Dict(conn => .2, lesion => .8),
                  n_perms::Int64=1000,
                  seed::Nullable{Int}=Nullable(1234))

  isnull(seed) || srand(get(seed))

  svrs::Dict{DataSet, PyObject} = Dict(conn => LinearSVR(C=conn_C),
                                       lesion => LinearSVR(C=lesion_C))

  both_ds(f::Function, T::Type) = Dict{DataSet, T}(
    [d => f(d)::T for d in [conn, lesion]])

  dis::Dict{DataSet, DataInfo} = both_ds(DataInfo) do d
    DataInfo(outcome, diff_wpm, all_subjects, region, d)
  end

  ids(di::DataInfo) = get_full(di)[:id]
  common_ids::Vector{UTF8String} = intersect(ids(dis[conn]), ids(dis[lesion]))
  common_id_ixs(di::DataInfo) = map(common_ids) do id
    ret::Vector{Int64} = find(ids(di) .== id)
    @assert length(ret) == 1
    ret[1]
  end

  typealias XY Tuple{Matrix{Float64}, Vector{Float64}}
  XYs::Dict{DataSet, XY} = both_ds(XY) do d
    di::DataInfo = dis[d]
    ixs::Vector{Int64} = common_id_ixs(di)
    (X::Matrix{Float64}, y::Vector{Float64}) = get_Xy_mat(di)
    X[ixs, :], y[ixs]
  end

  @assert XYs[conn][2] == XYs[lesion][2]

  fit(inds::Vector{Int64}) = both_ds(PyObject) do d
    svr::PyObject = svrs[d]
    (X::Matrix{Float64}, y::Vector{Float64}) = XYs[d]
    svr[:fit](X[inds, :], y[inds])
  end

  test(svrs::Dict{DataSet, PyObject}, inds::Vector{Int64}) = begin
    predictions::Matrix{Float64} = begin
      ret = zeros(Float64, length(inds), 2)
      for (ix::Int64, (d::DataSet, svr::PyObject)) in enumerate(svrs)
        X::Matrix{Float64} = XYs[d][1]
        ret[:, ix] = svr[:predict](X[inds, :]) .* weights[d]
      end
      ret
    end

    ensemble_predictions::Vector{Float64} = sum(predictions, 2)[:]
    @assert length(ensemble_predictions) == length(inds)

    y::Vector{Float64} = XYs[lesion][2]
    r2_score(y[inds], ensemble_predictions)
  end

  num_samples::Int64 = length(common_ids)

  cvg::CrossValGenerator = get_cvg(RandomSub, n_perms, num_samples)
  cross_validate(fit, test, num_samples, cvg)
end
