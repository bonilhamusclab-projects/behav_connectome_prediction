using DataFrames
using HypothesisTests
using Lazy
using MLBase
using PValueAdjust

include("DataInfo.jl")
include("helpers.jl")
include("svr_base.jl")


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
    DataFrame(
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

