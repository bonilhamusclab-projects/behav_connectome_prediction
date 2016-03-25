using DataFrames
using GLM
using Memoize
using PValueAdjust

include("helpers.jl")


@memoize function mk_edge_string(edge::Symbol)
  left, right = edge_col_to_roi_names(edge)
  "$(left) -- $(right)"
end


function create_edge_names(edges::Vector{Symbol})
  map(mk_edge_string, edges)
end


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


type DataTarget
  full::DataFrame
  target::Symbol
  covars::Vector{Symbol}
  rows_filter::Function
end


DataTarget(full::DataFrame, target::Symbol;
           covars::Vector{Symbol}=Symbol[],
           rows_filter::Function=df -> repmat([true], size(df, 1))) =
  DataTarget(full, target, covars, rows_filter)

Base.start(d::DataTarget) = 1
Base.getindex(d::DataTarget, s::Symbol) = d.(s)
Base.next(d::DataTarget, state) = d[fieldnames(DataTarget)[state]], state + 1
Base.done(d::DataTarget, state) = state > length(fieldnames(DataTarget))

function calc_all_measures(data_targets::Vector{DataTarget})

  ret = Dict()

  for (data, target, covars, rows_filter) in data_targets

    valid_rows::Vector{Bool} = rows_filter(data)
    data = data[valid_rows, :]

    function calc_sorted(edges::Vector{Symbol})
      sort(calc_coefs(data, target, edges, covars), cols=:pvalue)
    end

    edges = get_edges(data)

    k::Symbol = symbol(join([target; covars], "_cv_"))
    println(k)

    ret[k] = calc_sorted(edges)

    println("left")
    ret[symbol(k, "_left")] = calc_sorted(filter(is_left_hemi_edge, edges))

    println("left select")
    ret[symbol(k, "_left_select")] = calc_sorted(filter(is_left_hemi_select_edge, edges))
  end

  ret
end


function update_expr(e::Expr, old_term, new_term)
  parse(replace(
          string(e), old_term, new_term
          ))
end


macro calc_all_measures()
  quote
    function calc_dt_diff(m::MeasureGroup)
      full = @eval_str "full_$m()"
      target = @eval_str ":$(m)_diff_wpm"
      DataTarget(full, target, rows_filter=rows_filter_gen(m))
    end

    function calc_dt_covar(m::MeasureGroup)
      full = @eval_str "full_$m()"
      target = @eval_str ":se_$m"
      covars = @eval_str "[:pd_$m]"
      DataTarget(full, target, covars=covars, rows_filter = rows_filter_gen(m))
    end

    data_targets = [calc_dt_covar(adw), calc_dt_covar(atw),
                    calc_dt_diff(adw), calc_dt_diff(atw)]
    calc_all_measures(data_targets)
  end
end


function calc_scenario_all()
  rows_filter_gen(m::MeasureGroup) = d::DataFrame -> repmat([true], size(d, 1))

  @calc_all_measures
end


function calc_scenario_improved()
  @memoize diff_col(m::MeasureGroup) = symbol(m, "_diff_wpm")

  rows_filter_gen(m::MeasureGroup) =  d::DataFrame -> d[diff_col(m)] .> 0

  @calc_all_measures
end


function calc_scenario_poor_pd()
  @memoize pd_z_col(m::MeasureGroup) = symbol("pd_", m, "_z")

  rows_filter_gen(m::MeasureGroup) = d::DataFrame -> d[pd_z_col(m)] .< -1

  @calc_all_measures
end


function calc_all_scenarios()
  ret = Dict()
  ret[:all] = calc_scenario_all()
  ret[:improved] = calc_scenario_improved()
  ret[:poor_pd] = calc_scenario_poor_pd()
  ret
end


function save_calc_scenario_results(res::Dict, dir::AbstractString)
  for k in keys(res)
    df = res[k]::DataFrame
    writetable(joinpath("$dir", "$k.csv"), df)
  end
end

function save_all_scenarios(res::Dict)
  for (sk, sr) in res
    dir = joinpath("data/step4/scenarios/", "$sk")
    if !isdir(dir)
      mkpath(dir)
    end
    save_calc_scenario_results(sr, dir)
  end
end
