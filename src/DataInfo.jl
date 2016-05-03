include("helpers.jl")

immutable DataInfo
  outcome::Outcome
  target::Target
  subject_group::SubjectGroup
  region::Region
  dataset::DataSet
  include_covars::Bool
end

function DataInfo(outcome::Outcome, target::Target, subject_group::SubjectGroup,
                  region::Region, dataset::DataSet; include_covars::Bool=true)
  DataInfo(outcome, target, subject_group, region, dataset, include_covars)
end

getFull(d::DataInfo) = getFull(d.dataset, d.outcome)

getCovars(d::DataInfo) = d.include_covars ?
  getCovarsForTarget(d.target, d.outcome) : Symbol[]
getPredictors(d::DataInfo) = getPredictors(d.outcome, d.region, d.dataset)
getTargetCol(d::DataInfo) = getTargetCol(d.target, d.outcome)
getValidRows(d::DataInfo) = subjectFilter(d.subject_group,
                                             d.outcome,
                                             getFull(d))

getData(d::DataInfo) = begin
  target_col::Symbol = getTargetCol(d)
  covars::Vector{Symbol} = getCovars(d)
  predictors::Vector{Symbol} = getPredictors(d)
  valid_rows::Vector{Bool} = getValidRows(d)
  full::DataFrame = getFull(d)

  (full[valid_rows, [predictors; covars; target_col]], predictors, covars, target_col)
end


typealias XY Tuple{Matrix{Float64}, Vector{Float64}}
getXyMat(d::DataInfo) = begin
  data::DataFrame, predictors::Vector{Symbol}, covars::Vector{Symbol}, target_col::Symbol =
    getData(d)

  X::Matrix{Float64} = begin
    x_cols::Vector{Symbol} = [predictors; covars]
    Matrix(data[:, predictors])
  end

  y::Vector{Float64} = Array(data[:, target_col])

  X, y
end

to_string(d::DataInfo) = join(
  ["$(d.outcome)_$(d.subject_group)_$(d.region)_$(d.target)_$(d.dataset)"; getCovars(d)],
  "_cv_")

Base.start(d::DataInfo) = 1
Base.getindex(d::DataInfo, s::Symbol) = d.(s)
Base.next(d::DataInfo, state) = d[fieldnames(DataTarget)[state]], state + 1
Base.done(d::DataInfo, state) = state > length(fieldnames(DataTarget))

function forAllCombos(;fn::Function=(di::DataInfo) -> di,
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
