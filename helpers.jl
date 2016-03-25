using DataFrames
using Lazy
using Memoize

@enum MeasureGroup atw adw

function get_edges(df::DataFrame)
  filter(n -> startswith(string(n), "x"), names(df))
end

@memoize full_adw() = readtable("data/step3/full_adw.csv")
@memoize full_atw() = readtable("data/step3/full_atw.csv")

@memoize jhu() = readtable("data/jhu_coords.csv")

names_set(df_fn::Function) = @> df_fn()[:name] Set

@memoize jhu_left() = readtable("data/jhu_rois_left.csv")
@memoize jhu_left_names() = @> jhu_left names_set

@memoize jhu_left_select() = readtable("data/jhu_rois_left_adjusted.csv")
@memoize jhu_left_select_names() = @> jhu_left_select names_set

macro eval_str(s)
  quote
    eval(parse($s))
  end
end


edge_reg = r"x([0-9]+)_([0-9]+)"

@memoize function edge_col_to_x(edge::Symbol, x::Symbol)
  m = match(edge_reg, string(edge))
  map([1, 2]) do i
    ix = parse(Int64, m[i]) + 1
    jhu()[ix, x]
  end
end


@memoize function edge_col_to_roi_names(edge::Symbol)
  edge_col_to_x(edge, :name)
end


@memoize function is_left_hemi_edge(edge_col::Symbol)
  a, b = edge_col_to_roi_names(edge_col)
  in(a, jhu_left_names()) & in(b, jhu_left_names())
end


@memoize function is_left_hemi_select_edge(edge_col::Symbol)
  a, b = edge_col_to_roi_names(edge_col)
  in(a, jhu_left_select_names()) & in(b, jhu_left_select_names())
end
