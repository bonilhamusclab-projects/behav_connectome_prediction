using DataFrames
using HypothesisTests
using Lazy
using MLBase
using PValueAdjust

include("DataInfo.jl")
include("helpers.jl")
include("svrBase.jl")
include("ensembleSvr.jl")


function compareVectors{T <: AbstractVector{Float64}}(real::T, perm::T; ret::DataFrame=DataFrame())
  for fn in (mean, std)
    for (n, arr) in ((:real, real), (:perm, perm))
      ret[symbol("$(n)_$(fn)")] = [fn(arr)]
    end
  end

  diff_vec::AbstractVector{Float64} = isa(real, DataArray) ? dropna(real - perm) : real - perm
  ht::HtInfo = htInfo(diff_vec)
  for k::Symbol in keys(ht)
    ret[symbol("real_vs_perm_", k)] = [ht[k]]
  end

  ret
end


function compareCoefs(real_coefs::DataFrame, perm_coefs::DataFrame)

  name_coefs_map = Dict(:real => real_coefs, :perm => perm_coefs)

  predictors = names(real_coefs)
  @assert predictors == names(perm_coefs)

  calcRow(predictor::Symbol) = compareVectors(real_coefs[predictor],
                                              perm_coefs[predictor],
                                              ret=DataFrame(predictor=[predictor]))

  ret::DataFrame = vcat(map(calcRow, predictors)...)

  for m::Symbol in names(ret)
    is_p::Bool = endswith("$m", "_p")

    if is_p
      adj_measure = symbol(m, :adj)
      ret[adj_measure] = padjust(ret[m], Bonferroni)
    end
  end

  ret
end


compareCoefs(real_coefs::AbstractString, perm_coefs::AbstractString) = compareCoefs(
  readtable(real_coefs), readtable(perm_coefs))


function getDataDir!(d...)
  d_path = joinpath(d...)
  dir = joinpath(data_dir(), d_path)
  isdir(dir) || mkpath(dir)
  dir
end


@enum CompareType coefs scores
function _saveCompareRes(compares::DataFrame, suffix::AbstractString, compare_type::CompareType)
  dir = getDataDir!("step5", "svr")

  f_name = "real_vs_perm_$(compare_type)_$(suffix).csv"
  println(f_name)

  full_path = joinpath(dir, f_name)
  writetable(full_path, compares)
end


function saveCompareCoefs(coefCompares::DataFrame, suffix::AbstractString)
  _saveCompareRes(coefCompares, suffix, coefs)
end


compareScores(real_scores, perm_scores) = compareVectors(real_scores, perm_scores)

function compareScores(real_scores::AbstractString, perm_scores::AbstractString)
  compareVectors(readdlm(real_scores)[:], readdlm(perm_scores)[:])
end


function saveCompareScores(scoreCompares::DataFrame, suffix::AbstractString)
  _saveCompareRes(scoreCompares, suffix, scores)
end


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

  scores::Vector{Float64} = map(fitTest!, 1:num_repetitions)

  Dict(symbol(d, :_coefs) => coefs,
       symbol(d, :_scores) => scores)

end


function calcAllCoefs(di_state_map::Dict{DataInfo, State})

  ret = Dict{DataInfo, Dict{Symbol, Union{DataFrame, Vector{Float64}}}}()
  for (di::DataInfo, state::State) in di_state_map
    println(di)
    ret[di] = calcCoefs(di, state.cs) do rep_ix::Int64
      getIdIxs(di, state.getids(rep_ix))
    end
  end

  ret
end


function saveCalcCoefs(calc_all_coefs_ret::Dict{DataInfo, Dict{Symbol, Union{DataFrame, Vector{Float64}}}},
                       prefix::AbstractString)
  for di::DataInfo in keys(calc_all_coefs_ret)
    dir = getDataDir!("step4", "svr")
    for (k::Symbol, data::Union{Vector{Float64}, DataFrame}) in calc_all_coefs_ret[di]
      f_name = "$(prefix)_$k.csv"
      println(f_name)
      full_path = joinpath(dir, f_name)

      write_fn = isa(data, DataFrame) ? writetable : writecsv
      write_fn(full_path, data)
    end
  end
end


function saveCs(di_state_map::Dict{DataInfo, State}, prefix::AbstractString)
  for (di::DataInfo, s::State) in di_state_map
    saveStep4Array(s.cs, "$(prefix)_$(di)_cs")
  end
end


#Does not recalculate the coefficients
savePredScores(scores::AbstractArray{Float64},
              prefix::AbstractString) = saveStep4Array(scores,
  "$(prefix)_scores")


function saveStep4Array(arr::AbstractArray{Float64}, f_name)
  dir = getDataDir!("step4", "svr")
  full_path = joinpath(dir, "$(f_name).csv")
  writecsv(full_path, arr)
end
