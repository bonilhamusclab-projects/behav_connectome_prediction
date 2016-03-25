using DataFrames
using Lazy
using Memoize

@enum MeasureGroup atw adw

@enum Target diff_wpm se

@enum Region left_select left full_brain

@enum SubjectGroup poor_pd improved all_subjects

@memoize function is_left_hemi_edge(edge_col::Symbol)
  a, b = edge_col_to_roi_names(edge_col)
  in(a, jhu_left_names()) & in(b, jhu_left_names())
end


@memoize function is_left_hemi_select_edge(edge_col::Symbol)
  a, b = edge_col_to_roi_names(edge_col)
  in(a, jhu_left_select_names()) & in(b, jhu_left_select_names())
end


@memoize function region_filter_gen(region::Region)
  @switch region begin
    left_select; is_left_hemi_select_edge
    left; is_left_hemi_edge
    full_brain; edge::Symbol -> true
  end
end


function get_edges(df::DataFrame; region::Region=full_brain)
  edge_filter(n::Symbol) = startswith(string(n), "x")
  region_filter = region_filter_gen(region)

  filter(n::Symbol -> edge_filter(n) && region_filter(n), names(df))
end


function get_edges(m::MeasureGroup, region::Region)
  full::DataFrame = @eval_str "full_$m()"
  get_edges(full, region=region)
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


@memoize function mk_edge_string(edge::Symbol)
  left, right = edge_col_to_roi_names(edge)
  "$(left) -- $(right)"
end


function create_edge_names(edges::Vector{Symbol})
  map(mk_edge_string, edges)
end


@memoize function subject_filter_gen_gen(s::SubjectGroup)
  @switch s begin
    all_subjects; m::MeasureGroup -> d::DataFrame -> repmat([true], size(d, 1))
    improved; m::MeasureGroup -> d::DataFrame -> d[symbol(m, "_diff_wpm")] .> 0
    poor_pd; m::MeasureGroup -> d::DataFrame -> d[symbol("pd_", m, "_z")] .< 0
  end
end
