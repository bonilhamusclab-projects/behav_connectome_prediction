using DataFrames

function convertLesionsToNodes(lesion_f::AbstractString)
    lesion_df::DataFrame = readtable(lesion_f)
end