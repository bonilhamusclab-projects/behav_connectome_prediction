using DataFrames
using HypothesisTests
using Lazy
using MLBase
using PValueAdjust

include("DataInfo.jl")
include("helpers.jl")
include("svrBase.jl")
include("ensembleSvr.jl")


typealias IndicesLookup Array{Array{Int64}}

function calcCoefs(get_ixs::Function,
                  d::DataInfo,
                  Cs::AbstractVector{Float64})

  X::Matrix{Float64}, y::Vector{Float64} = getXyMat(d)

  num_repetitions = length(Cs)

  predictors = getPredictors(d)
  n_predictors = length(predictors)

  coefs::DataFrame = begin
    ret = DataFrame()
    for p in predictors
      ###Hack to keep it float64 while being NA
      ret[p] = repmat([-Inf], num_repetitions)
      ret[ret[p] .== -Inf, :] = NA
    end
    ret
  end

  fitTest! = begin
    svr = LinearSVR()

    function updateCoefs!(repetition_ix::Int64, coef_data::Vector{Float64})
      for (ix, c) in enumerate(coef_data)
        coefs[repetition_ix, predictors[ix]] = c
      end
    end

    typealias XyIxs Tuple{Vector{Int64}, Vector{Int64}}
    typealias TrainTestIxs Tuple{XyIxs, XyIxs}

    function fn(repetition_ix::Int64)
      println(repetition_ix)

      svr[:C] = Cs[repetition_ix]

      ixs::TrainTestIxs = get_ixs(repetition_ix)

      x_train_ixs, y_train_ixs = ixs[1]
      svr[:fit](X[x_train_ixs, :], y[y_train_ixs])

      updateCoefs!(repetition_ix, svr[:coef_])

      x_test_ixs, y_test_ixs = ixs[2]
      r2Score(y[y_test_ixs], svr[:predict](X[x_test_ixs, :]))
    end
  end

  test_scores::Vector{Float64} = map(fitTest!, 1:num_repetitions)

  pv(arr::Vector{Float64}, tail::Symbol=:both) = pvalue(OneSampleTTest(arr), tail=tail)

  prediction_info::DataFrame = begin
    ht = htInfo(test_scores)
    DataFrame(
      mean=mean(test_scores),
      std=std(test_scores),
      t=ht[:t],
      right_p=ht[:right_p],
      left_p=ht[:left_p],
      both_p=ht[:both_p],
      num_repetitions=num_repetitions)
  end

  predictor_info::DataFrame = begin
    pr = DataFrame()
    pr[symbol(d.dataset)] = predictors
    pr[symbol(d.dataset, :_name)] = createPredictorNames(predictors, d.dataset)


    pred_apply(fn::Function) = [fn(dropna(coefs[p])) for p in predictors]

    pr[:mean] = pred_apply(mean)
    pr[:std] = pred_apply(std)

    hts::Vector{HtInfo} = pred_apply(htInfo)
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


function calcAllCoefs(di_state_map::Dict{DataInfo, State})

  ret = Dict{DataInfo, Dict{Symbol, DataFrame}}()
  for (di::DataInfo, state::State) in di_state_map
    println(di)
    ret[di] = calcCoefs(di, state.cs) do rep_ix::Int64
      getIdIxs(di, state.getids(rep_ix))
    end
  end

  ret
end


function saveCalcCoefs(calc_all_coefs_ret::Dict{DataInfo, Dict{Symbol, DataFrame}})
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
