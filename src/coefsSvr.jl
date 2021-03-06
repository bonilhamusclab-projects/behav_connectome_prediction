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


function calcCoefs(di::DataInfo,
                  getixs::Function,
                  ids::Ids,
                  Cs::AbstractVector{Float64})

  X::Matrix{Float64}, y::Vector{Float64} = getXyMat(di, ids)
  pipe = svrPipelineGen(X, y)

  num_repetitions = length(Cs)

  predictors = getPredictors(di)
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
    function updateCoefs!(repetition_ix::Int64, coef_data::Vector{Float64})
      for (ix, c) in enumerate(coef_data)
        coefs[repetition_ix, predictors[ix]] = c
      end
    end

    typealias XyIxs Tuple{Vector{Int64}, Vector{Int64}}
    typealias TrainTestIxs Tuple{XyIxs, XyIxs}

    function fn(repetition_ix::Int64)
      println("repetition index: $(repetition_ix)")

      paramState(pipe, :svr)[:C] = Cs[repetition_ix]

      ixs::TrainTestIxs = getixs(repetition_ix)

      x_train_ixs, y_train_ixs = ixs[1]
      pipeFit!(pipe, x_train_ixs, y_train_ixs)

      updateCoefs!(repetition_ix, paramState(pipe, :svr)[:coef_])

      x_test_ixs, y_test_ixs = ixs[2]
      pipeTest(pipe, x_test_ixs)
    end
  end

  scores::Vector{Float64} = map(fitTest!, 1:num_repetitions)

  Dict(symbol(di, :_coefs) => coefs,
       symbol(di, :_scores) => scores)

end


function calcAllCoefs(di_state_map::Dict{DataInfo, State}, ids, getrepixs)

  ret = Dict{DataInfo, Dict{Symbol, Union{DataFrame, Vector{Float64}}}}()
  for (di::DataInfo, state::State) in di_state_map
    println(di)
    ret[di] = calcCoefs(di, getrepixs, ids, state.cs)
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


function saveEnsembleAnalysis(analysis::Dict, name::AbstractString)
  saveCs(analysis[:state], name)
  savePredScores(analyis[:preds][55], "$(name)_left_select2_ensemble")

  coefs = calcAllCoefs(analysis[:state], analysis[:ids], analysis[:getrepixs])
  saveCalcCoefs(coefs, name)
end


function calcPermCoefs(di::DataInfo,
  ids::Ids,
  cs::AbstractString,
  scores::AbstractString,
  perm_coefs_df::AbstractString)

  cs_arr, scores_arr = @>> (cs, scores) map(f -> readcsv(f, Float64)[:])
  calcPermCoefs(di, ids, cs_arr, scores_arr, readtable(perm_coefs_df))
end


function calcPermCoefs(di, ids, cs::Vector, scores::Vector, perm_coefs_df)
  num_cs = length(cs)
  @assert num_cs == length(scores)

  weighted_c = dot(cs, scores)/sum(scores)
  println("c: $(weighted_c)")

  ixs = 1:length(ids)

  getrepixs(i) = ((ixs, ixs), (ixs, ixs))

  coefs_df::DataFrame = begin
    coefs_analysis = calcCoefs(di, getrepixs, ids, [weighted_c])
    di_coefs::Symbol = symbol(di, "_coefs")

    di_scores::Symbol = symbol(di, "_scores")
    println("scores: $(coefs_analysis[di_scores])")

    coefs_analysis[di_coefs]
  end

  function getP(n::Symbol)
    diff = coefs_df[n] .- perm_coefs_df[n]

    null_gt_perm = @>> diff OneSampleTTest pvalue(tail=:left)
    null_eq_perm = @>> diff OneSampleTTest pvalue
    null_lt_perm = @>> diff OneSampleTTest pvalue(tail=:right)

    [null_gt_perm, null_eq_perm, null_lt_perm]
  end

  pvalues = [n => getP(n) for n in names(coefs_df)] |> DataFrame

  pvalues_adj = begin
    predictors = names(pvalues)

    @assert size(coefs_df) == (1, length(predictors))

    perm_mean = @> perm_coefs_df Matrix mean(1)
    @assert size(perm_mean) == size(coefs_df)

    perm_std = @> perm_coefs_df Matrix std(1)

    ret = DataFrame(predictor=predictors,
      coef = Matrix(coefs_df)[:],
      perm_mean = perm_mean[:],
      perm_std = perm_std[:])

    pvalues_mat = Matrix(pvalues)

    for (i, ptype) in enumerate([:null_gt_perm, :null_eq_perm, :null_lt_perm])
      pvalues_arr::Vector{Float64} = pvalues_mat[i, :][:]
      adj_arr = padjust(pvalues_arr, Bonferroni)

      ret[symbol(ptype, "_adj")] = adj_arr
      ret[ptype] = pvalues_arr
    end

    ret
  end

  pvalues_adj
end
