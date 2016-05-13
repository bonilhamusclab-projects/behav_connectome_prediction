using DataFrames
using GLM
using Memoize
using PValueAdjust

include("DataInfo.jl")
include("helpers.jl")


macro update_covar(k, v)
  :(
    covar_dt[symbol(c, "_", $k)][ix] = $v
  )
end


function calcCoefs(df::DataFrame, target::Symbol, predictors::Vector{Symbol},
                    covars::Vector{Symbol}=Symbol[])

  num_predictors = length(predictors)

  mkZeros = () -> zeros(Float64, num_predictors)

  coefs = mkZeros()
  cors = mkZeros()
  pvalues = ones(Float64, num_predictors)
  tscores = mkZeros()

  num_covars = length(covars)
  covar_dt = begin
    mkOnes = () -> ones(Float64, num_predictors)
    ret = Dict{Symbol, Vector{Float64}}()
    for c in covars
      for (t, fn) in (("coef", mkZeros), ("pvalue", mkOnes), ("tscore", mkZeros))
        k::Symbol = symbol(c, "_", t)
        ret[k] = fn()
      end
    end

    ret
  end

  for (ix, predictor) = enumerate(predictors)
    if all(df[predictor] .== 0.)
      continue
    end
    fm_str = "$target ~ $predictor"
    if !isempty(covars)
      fm_str = string(fm_str, " + ", join(covars, " + "))
    end

    fm = eval(parse(fm_str))
    ct = coeftable(lm(fm, df))
    coefs[ix], tscores[ix], pvalues[ix] = ct.mat[2, [1, 3, 4]]

    for c in covars
      c_c, c_t, c_p = ct.mat[3, [1, 3, 4]]
      @update_covar :coef c_c
      @update_covar :tscore c_t
      @update_covar :pvalue c_p
    end

    cors[ix] = cor(df[predictor], df[target])
  end

  left_p_values, right_p_values = begin
    one_sided_pvalues = pvalues./2
    most_area_pvalues = 1 - one_sided_pvalues
    take_left = tscores .< 0
    take_right = !take_left
    (take_left .* one_sided_pvalues + take_right .* most_area_pvalues,
    take_right .* one_sided_pvalues + take_left .* most_area_pvalues)
  end

  ret = DataFrame(coef=coefs, pvalue=pvalues,
    pvalue_left=left_p_values, pvalue_right=right_p_values,
    tscore=tscores, predictor=predictors, cor_coef=cors)

  for p::Symbol in (:pvalue, :pvalue_left, :pvalue_right)
    ret[symbol(p, "_adj")] = padjust(ret[p], BenjaminiHochberg)
  end

  for c in keys(covar_dt)
    ret[c] = covar_dt[c]
    if contains("$c", "pvalue")
      ret[symbol(c, "_adj")] = padjust(ret[c], BenjaminiHochberg)
    end
  end

  ret
end


function calcAllMeasures(data_targets::Vector{DataInfo})

  ret = Dict()

  for d::DataInfo in data_targets
    println(d)

    data::DataFrame, edges::Vector{Symbol}, covars::Vector{Symbol}, target_col::Symbol =
      getData(d)

    coefs = calcCoefs(data, target_col, edges, covars)

    ret[d] = sort(coefs, cols=:pvalue)
  end

  ret
end


macro calc_all_measures()
  quote
    data_targets::Vector{DataInfo} =
      [DataInfo(o, t, s, r, d)
       for o::Outcome in (atw, adw),
       t::Target in (se, diff_wpm),
       r::Region in (full_brain, left, left_select),
       d::DataSet in (conn, lesion)
       ][:]
    calcAllMeasures(data_targets)
  end
end


function calcAllSubjects()
  s::SubjectGroup= all_subjects

  @calc_all_measures
end


function calcImprovedSubjects()
  s::SubjectGroup = improved

  @calc_all_measures
end


function calcPoorPdSubjects()
  s::SubjectGroup = poor_pd

  @calc_all_measures
end


calcAllScenarios() = Dict(
  union(calcAllSubjects(), calcImprovedSubjects(), calcPoorPdSubjects())
)


saveCalcScenarioResults(res::Dict, dir::AbstractString) = for di::DataInfo in keys(res)
  df = res[di]::DataFrame
  writetable(joinpath(dir, "$(di).csv"), df)
end


function saveAllScenarios(res::Dict)
  dir = "../data/step4/linear_reg/"
  isdir(dir) || mkpath(dir)

  saveCalcScenarioResults(res, dir)
end
