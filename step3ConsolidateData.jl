using DataFrames
using Memoize


loadRoiNames() = Vector{Symbol}(map(symbol, readtable("data/jhu_coords.csv")[:name]))


@memoize getRoiIx(roi::Symbol) = findfirst(e -> e == roi, loadRoiNames())


getId(csv_path::AbstractString) = basename(csv_path)[1:end-4]


function loadConnCsv(csv_path, rois=loadRoiNames())
  roi_x_roi = readtable(csv_path, header=false, names=rois)

  num_rois = length(rois)

  @assert size(roi_x_roi) == (num_rois, num_rois)

  roi_x_roi[:left] = rois
  roi_long = rename!(melt(roi_x_roi, :left), :variable, :right)
  roi_long[:left_ix] = map(getRoiIx, roi_long[:left])
  roi_long[:right_ix] = map(getRoiIx, roi_long[:right])

  roi_long = roi_long[roi_long[:left_ix] .> roi_long[:right_ix], [:left, :right, :value]]
  conn_names::Vector{Symbol} = map(eachrow(roi_long)) do r
    symbol(r[:left], "_to_" , r[:right])
  end
  conn_values = Any[[v] for v in roi_long[:value]]

  ret = DataFrame(conn_values, conn_names)
  ret[:id] = getId(csv_path)
  ret
end


function loadLesionCsv(csv_path, rois=loadRoiNames())
  data::Vector{Any} = Any[[i] for i in readcsv(csv_path)]
  @assert length(rois) == length(data)

  ret = DataFrame(data, rois)
  ret[:id] = getId(csv_path)
  ret
end


typealias OptDir Nullable{ASCIIString}


function consolidateX(load_fn::Function,
                       csv_dir::AbstractString;
                       dest_dir::OptDir = OptDir())
  fs = readdir(csv_dir)
  num_fs = length(fs)

  curr_load::Int64 = 0
  function load(csv::AbstractString)
    curr_load += 1
    println("$(curr_load) out of $(num_fs)")
    load_fn(joinpath(csv_dir, csv))
  end

  consolidated_df::DataFrame = vcat(DataFrame[load(c) for c in fs])
  @assert size(consolidated_df, 1) == num_fs

  if !isnull(dest_dir)
    dest_path = joinpath(get(dest_dir), "X.csv")
    writetable(dest_path, consolidated_df)
  end

  consolidated_df
end


function consolidateConnX(csv_dir="data/step1/conn";
                            dest_dir::OptDir=OptDir())
  println("conn")
  rois = loadRoiNames()

  consolidateX(csv_dir, dest_dir=dest_dir) do csv_path
    loadConnCsv(csv_path, rois)
  end
end


function consolidateLesionX(csv_dir="data/step1/lesion";
                              dest_dir::OptDir=OptDir())
  println("lesion")
  rois::Vector{Symbol} = loadRoiNames()

  consolidateX(loadLesionCsv, csv_dir, dest_dir=dest_dir)
end


function calcWpsDiff!(y::DataFrame)
  second_look_up = Dict(:elvis_dw => 38, :mlk_dw => 44, :sm_dw => 43)
  num_rows = size(y, 1)

  naZeros(k::Symbol, v) = [isna(i) ? 0. : v for i in y[k]]
  naZeros(k::Symbol) = [isna(i) ? 0. : i for i in y[k]]

  total_seconds = reduce(zeros(num_rows), keys(second_look_up)) do acc, k
    acc .+ naZeros(k, second_look_up[k])
  end

  y[:se_adw_wps] = sum([naZeros(k) for k in keys(second_look_up)])./total_seconds

  y[:pd_adw_wps] = y[:pd_adw]/120.

  y[:adw_diff_wps] = y[:se_adw_wps] - y[:pd_adw_wps]

  y
end


function consolidateXwithY(consolidated_x::DataFrame,
                              y_dir="data/step2/";
                              dest_dir::OptDir=OptDir())

  @assert in(:id, names(consolidated_x))

  ret = Dict()
  meta = readtable(joinpath(y_dir, "meta.csv"))

  idjoin(a, b) = join(a, b, on=:id)

  for yo in ["adw", "atw"]
    y_src = joinpath(y_dir, "$(yo)_outcomes.csv")
    y = readtable(y_src)
    (yo == "adw") && calcWpsDiff!(y)
    
    full = begin
      xy = idjoin(consolidated_x, y)
      idjoin(xy, meta)
    end

    max_num_rows = min(size(consolidated_x, 1), size(y, 1))
    num_cols = size(consolidated_x, 2) + size(y, 2) + size(meta, 2) - 2

    @assert size(full, 1) <= max_num_rows
    @assert size(full, 2) == num_cols

    if !isnull(dest_dir)
      dest_path = joinpath(get(dest_dir), "full_$(yo).csv")
      writetable(dest_path, full)
    end

    ret[yo] = full
  end

  ret
end


function runAll()
  lesion_dir = "data/step3/lesion"
  conn_dir = "data/step3/conn"

  X_lesion = consolidateLesionX(dest_dir=OptDir(lesion_dir))
  consolidateXwithY(X_lesion, dest_dir=OptDir(lesion_dir))

  X_conn = consolidateConnX(dest_dir=OptDir(conn_dir))
  consolidateXwithY(X_conn, dest_dir=OptDir(conn_dir))
end
