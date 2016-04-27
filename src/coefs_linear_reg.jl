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

function calc_coefs(df::DataFrame, target::Symbol, edges::Vector{Symbol},
                    covars::Vector{Symbol}=Symbol[])

  num_edges = length(edges)

  mk_zeros = () -> zeros(Float64, num_edges)

  coefs = mk_zeros()
  cors = mk_zeros()
  pvalues = ones(Float64, num_edges)
  tscores = mk_zeros()

  num_covars = length(covars)
  covar_dt = begin
    mk_ones = () -> ones(Float64, num_edges)
    ret = Dict{Symbol, Vector{Float64}}()
    for c in covars
      for (t, fn) in (("coef", mk_zeros), ("pvalue", mk_ones), ("tscore", mk_zeros))
        k::Symbol = symbol(c, "_", t)
        ret[k] = fn()
      end
    end

    ret
  end

  for (ix, edge) = enumerate(edges)
    if all(df[edge] .== 0.)
      continue
    end
    fm_str = "$target ~ $edge"
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

    cors[ix] = cor(df[edge], df[target])
  end

  edge_names = create_edge_names(edges)

  ret = DataFrame(coef=coefs, pvalue=pvalues, tscore=tscores, edge=edges, edge_name=edge_names, cor_coef=cors)
  ret[:pvalue_adj] = padjust(ret[:pvalue], BenjaminiHochberg)

  for c in keys(covar_dt)
    ret[c] = covar_dt[c]
    if contains("$c", "pvalue")
      ret[symbol(c, "_adj")] = padjust(ret[c], BenjaminiHochberg)
    end
  end

  ret
end


function calc_all_measures(data_targets::Vector{DataInfo})

  ret = Dict()

  for d::DataInfo in data_targets
    d_string = to_string(d)
    println(d_string)

    data::DataFrame, edges::Vector{Symbol}, covars::Vector{Symbol}, target_col::Symbol =
      get_data(d)

    coefs = calc_coefs(data, target_col, edges, covars)

    ret[symbol(d_string)] = sort(coefs, cols=:pvalue)
  end

  ret
end


macro calc_all_measures()
  quote
    data_targets::Vector{DataInfo} =
      [DataInfo(m, t, s, r)
       for m::Outcome in (atw, adw),
       t::Target in (se, diff_wpm),
       r::Region in (full_brain, left, left_select)][:]
    calc_all_measures(data_targets)
  end
end


function calc_all_subjects()
  s::SubjectGroup= all_subjects

  @calc_all_measures
end


function calc_improved_subjects()
  s::SubjectGroup = improved

  @calc_all_measures
end


function calc_poor_pd_subjects()
  s::SubjectGroup = poor_pd

  @calc_all_measures
end


function calc_all_scenarios()
  ret = Dict()
  ret[all_subjects] = calc_all_subjects()
  ret[improved] = calc_improved_subjects()
  ret[poor_pd] = calc_poor_pd_subjects()
  ret
end


function save_calc_scenario_results(res::Dict, dir::AbstractString)
  for k in keys(res)
    df = res[k]::DataFrame
    writetable(joinpath("$dir", "$k.csv"), df)
  end
end

function save_all_scenarios(res::Dict)
  for (s::SubjectGroup, sr) in res
    dir = joinpath("data/step4/linear_reg/", "$s")
    isdir(dir) || mkpath(dir)

    save_calc_scenario_results(sr, dir)
  end
end