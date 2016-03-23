using DataFrames
using GLM
using Memoize
using PValueAdjust

full_adw = readtable("data/step3/full_adw.csv")
full_atw = readtable("data/step3/full_atw.csv")

jhu = readtable("data/jhu_coords.csv")

jhu_left = readtable("data/jhu_rois_left.csv")
jhu_left_names = Set(jhu_left[:name])

jhu_left_select = readtable("data/jhu_rois_left_adjusted.csv")
jhu_left_select_names = Set(jhu_left_select[:name])

function get_edges(df::DataFrame)
  filter(n -> startswith(string(n), "x"), names(df))
end

edge_reg = r"x([0-9]+)_([0-9]+)"

@memoize function edge_col_to_x(edge::Symbol, x::Symbol)
  m = match(edge_reg, string(edge))
  map([1, 2]) do i
    ix = parse(Int64, m[i]) + 1
    jhu[ix, x]
  end
end

@memoize function edge_col_to_roi_names(edge::Symbol)
  edge_col_to_x(edge, :name)
end

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

@memoize function is_left_hemi_edge(edge_col::Symbol)
  a, b = edge_col_to_roi_names(edge_col)
  in(a, jhu_left_names) & in(b, jhu_left_names)
end

@memoize function is_left_hemi_select_edge(edge_col::Symbol)
  a, b = edge_col_to_roi_names(edge_col)
  in(a, jhu_left_select_names) & in(b, jhu_left_select_names)
end

function calc_all()

  data_targets = [(full_adw, :se_adw, [:pd_adw]), (full_atw, :se_atw, [:pd_atw])]

  ret = Dict()

  for (data, target, covars) = data_targets

    function calc_sorted(edges::Vector{Symbol})
      sort(calc_coefs(data, target, edges, covars), cols=:pvalue_adj)
    end

    edges = get_edges(data)
    println(target)

    ret[target] = calc_sorted(edges)

    println("left")
    ret[symbol(target, "_left")] = calc_sorted(filter(is_left_hemi_edge, edges))

    println("left select")
    ret[symbol(target, "_left_select")] = calc_sorted(filter(is_left_hemi_select_edge, edges))
  end

  ret

end

function save_calc_all_results(res::Dict)
  for k in keys(res)
    df = res[k]::DataFrame
    writetable("data/step4/valid_coefs_explore/$k.csv", df)
  end
end
