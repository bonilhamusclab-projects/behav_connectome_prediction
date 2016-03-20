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

function calc_coefs(df::DataFrame, target::Symbol, edges::Vector{Symbol})

  num_edges = length(edges)

  mk_zeros = () -> zeros(Float64, num_edges)

  coefs = mk_zeros()
  cors = mk_zeros()
  pvalues = ones(Float64, num_edges)
  tscores = mk_zeros()

  for (ix, edge) = enumerate(edges)
    if all(df[edge] .== 0.)
      continue
    end
    fm_str = "$target ~ $edge"
    fm = eval(parse(fm_str))
    ct = coeftable(lm(fm, df))
    coefs[ix], tscores[ix], pvalues[ix] = ct.mat[2, [1, 3, 4]]
    cors[ix] = cor(df[edge], df[target])
  end

  edge_names = create_edge_names(edges)
  ret = DataFrame(coef=coefs, pvalue=pvalues, tscore=tscores, edge=edges, edge_name=edge_names, cor_coef=cors)

  ret[:pvalue_adj] = padjust(ret[:pvalue], BenjaminiHochberg)

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

  data_targets = [(full_adw, :adw_z), (full_atw, :atw_diff_wpm), (full_atw, :atw_z)]

  ret = Dict()

  for (data, target) = data_targets

    function calc_sorted(data::DataFrame, edges::Vector{Symbol})
      sort(calc_coefs(data, target, edges), cols=:pvalue_adj)
    end

    println(target)
    edges = get_edges(data)
    ret[target] = calc_sorted(data, get_edges(data))
    println("left")
    ret[symbol(target, "_left")] = calc_sorted(data, filter(is_left_hemi_edge, edges))
    println("left select")
    ret[symbol(target, "_left_select")] = calc_sorted(data, filter(is_left_hemi_select_edge, edges))
  end

  ret

end

function save_calc_all_results(res::Dict)
  for k in keys(res)
    df = res[k]::DataFrame
    writetable("data/step4/valid_coefs_explore/$k.csv", df)
  end
end
