include("helpers.jl")

type DataInfo
  measure_group::MeasureGroup
  target::Target
  subject_group::SubjectGroup
  region::Region
end

get_full(d::DataInfo) = @eval_str "full_$(d.measure_group)()"

get_covars(d::DataInfo) = covars_for_target(d.target, d.measure_group)
get_edges(d::DataInfo) = get_edges(d.measure_group, d.region)
get_target_col(d::DataInfo) = get_target_col(d.target, d.measure_group)
get_valid_rows(d::DataInfo) = subject_filter(
  d.subject_group, d.measure_group, get_full(d))

get_data(d::DataInfo) = begin
  target_col::Symbol = get_target_col(d)
  covars::Vector{Symbol} = get_covars(d)
  edges::Vector{Symbol} = get_edges(d)
  valid_rows::Vector{Bool} = get_valid_rows(d)
  full::DataFrame = get_full(d)

  (full[valid_rows, [edges; covars; target_col]], edges, covars, target_col)
end

get_Xy_mat(d::DataInfo) = begin
  data::DataFrame, edges::Vector{Symbol}, covars::Vector{Symbol}, target_col::Symbol =
    get_data(d)

  X::Matrix{Float64} = begin
    x_cols::Vector{Symbol} = [edges; covars]
    Matrix(data[:, edges])
  end

  y::Vector{Float64} = Array(data[:, target_col])

  X, y
end

to_string(d::DataInfo) = join(
  ["$(d.measure_group)_$(d.subject_group)_$(d.region)_$(d.target)"; get_covars(d)],
  "_cv_")

Base.start(d::DataInfo) = 1
Base.getindex(d::DataInfo, s::Symbol) = d.(s)
Base.next(d::DataInfo, state) = d[fieldnames(DataTarget)[state]], state + 1
Base.done(d::DataInfo, state) = state > length(fieldnames(DataTarget))
