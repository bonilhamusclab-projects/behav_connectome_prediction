include("helpers.jl")

immutable DataInfo
  outcome::Outcome
  target::Target
  subject_group::SubjectGroup
  region::Region
  dataset::DataSet
end

get_full(d::DataInfo) = get_full(d.dataset, d.outcome)

get_covars(d::DataInfo) = covars_for_target(d.target, d.outcome)
get_predictors(d::DataInfo) = get_predictors(d.outcome, d.region, d.dataset)
get_target_col(d::DataInfo) = get_target_col(d.target, d.outcome)
get_valid_rows(d::DataInfo) = subject_filter(d.subject_group,
                                             d.outcome,
                                             get_full(d))

get_data(d::DataInfo) = begin
  target_col::Symbol = get_target_col(d)
  covars::Vector{Symbol} = get_covars(d)
  predictors::Vector{Symbol} = get_predictors(d)
  valid_rows::Vector{Bool} = get_valid_rows(d)
  full::DataFrame = get_full(d)

  (full[valid_rows, [predictors; covars; target_col]], predictors, covars, target_col)
end

get_Xy_mat(d::DataInfo) = begin
  data::DataFrame, predictors::Vector{Symbol}, covars::Vector{Symbol}, target_col::Symbol =
    get_data(d)

  X::Matrix{Float64} = begin
    x_cols::Vector{Symbol} = [predictors; covars]
    Matrix(data[:, predictors])
  end

  y::Vector{Float64} = Array(data[:, target_col])

  X, y
end

to_string(d::DataInfo) = join(
  ["$(d.outcome)_$(d.subject_group)_$(d.region)_$(d.target)_$(d.dataset)"; get_covars(d)],
  "_cv_")

Base.start(d::DataInfo) = 1
Base.getindex(d::DataInfo, s::Symbol) = d.(s)
Base.next(d::DataInfo, state) = d[fieldnames(DataTarget)[state]], state + 1
Base.done(d::DataInfo, state) = state > length(fieldnames(DataTarget))

function for_all_combos(;fn::Function=(di::DataInfo) -> di,
                        outcomes::Vector{Outcome}=Outcome[adw, atw],
                        subject_groups::Vector{SubjectGroup}=SubjectGroup[all_subjects, improved, poor_pd, poor_pd_1],
                        targets::Vector{Target}=[diff_wpm, se],
                        regions::Vector{Region}=[full_brain, left_select, left],
                        datasets::Vector{DataSet}=[conn, lesion])

  ret = DataInfo[]

  for o in outcomes
    for t in targets
      for s in subject_groups
        for r in regions
          for d in datasets
            di::DataInfo = fn(DataInfo(o, t, s, r, d))
            push!(ret, di)
          end
        end
      end
    end
  end

  sort(ret)

end


Base.isless(x::DataInfo, y::DataInfo) = begin
  for f in fieldnames(x)
    if x.(f) != y.(f)
      return x.(f) < y.(f)
    end
  end

  false
end

Base.show(io::IO, di::DataInfo) = print(io, to_string(di))
