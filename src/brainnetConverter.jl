using DataFrames

include("helpers.jl")

takeColGen(col::Symbol, fn::Function=identity) = df::DataFrame -> fn(df[col])


function getCoords(names::Vector{ASCIIString})
    jhu_df::DataFrame = jhu()
    joined::DataFrame = join(DataFrame(name=names), jhu_df, on=:name)
    joined[:x], joined[:y], joined[:z]
end


function convertLesionsToNodes(lesion_f::AbstractString, 
                               sizes_fn::Function,
                               names_fn::Function,
                               colors_fn::Function)
    
    
    lesion_df::DataFrame = readtable(lesion_f) 
    sizes::Vector{Float64} = Float64[isnan(s) ? 0.0 : s for s in sizes_fn(lesion_df)]
    
    names::Vector{ASCIIString} = names_fn(lesion_df)
    typealias Reals AbstractVector{Union{Float64, Int64}}
    colors::Reals = colors_fn(lesion_df)
    
    typealias Points AbstractVector{Float64}
    x_coords::Points, y_coords::Points, z_coords::Points = getCoords(names)
    
    DataFrame(x=x_coords, y=y_coords, z=z_coords, color=colors, size=sizes, name=names)
end


function convertLesionsToNodes(lr_lesion_f::AbstractString, svr_lesion_f::AbstractString)
    normalize(f::AbstractVector) = f./maximum(abs(f))
    colorsFnGen(col::Symbol) = takeColGen(col, sign)
    sizesFnGen(col::Symbol) = takeColGen(col, f::AbstractVector -> abs(normalize(f)))
    
    lr_nodes = convertLesionsToNodes(lr_lesion_f,
        sizesFnGen(:tscore), takeColGen(:edge), colorsFnGen(:tscore))
        
    svr_nodes = convertLesionsToNodes(svr_lesion_f,
        sizesFnGen(:t), takeColGen(:lesion), colorsFnGen(:t))
        
    lr_nodes, svr_nodes
end


function convertConnectionsToEdges(connection_df::DataFrame,
                                   names_col::Symbol,
                                   weights_col::Symbol,
                                   split_key::ASCIIString,
                                   node_names::AbstractVector{ASCIIString};
                                   is_symmetric::Bool=true)
     conn_names::AbstractVector{ASCIIString} = connection_df[names_col]
     
     conn_matrix::Matrix{Real} = begin
        num_nodes = length(node_names)
        zeros(Real, num_nodes, num_nodes)
     end
     
     for conn_name in conn_names
        left_ix::Nullable{Int64}, right_ix::Nullable{Int64} = map(split(conn_name, split_key)) do node_name::AbstractString 
            locs::Vector{Int64} = findin(node_names, [node_name])
            length(locs) == 0 ? Nullable{Int64}() : Nullable(locs[1])
        end
        
        if !(isnull(left_ix) || isnull(right_ix))
            i::Int64, j::Int64 = get(left_ix), get(right_ix)
            weight::Real = connection_df[connection_df[names_col] .== conn_name, weights_col][1]
            conn_matrix[i, j] = weight
            is_symmetric && (conn_matrix[j, i] = weight)
        end
        
     end
     
     conn_matrix
end


function convertConnectionsToEdges(lr_conn_f::AbstractString, svr_conn_f::AbstractString,
                                   lr_nodes::DataFrame, svr_nodes::DataFrame)
     lr_edges::Matrix{Real} = convertConnectionsToEdges(
         readtable(lr_conn_f), :edge, :tscore, "_to_", lr_nodes[:name]);
     
     svr_edges::Matrix{Real} = convertConnectionsToEdges(
         readtable(svr_conn_f), :conn, :t, "_to_", svr_nodes[:name]);
         
     function originalToConvenince(original_edges::Matrix{Real})
        function zeroOut(ixs)
            ret = copy(original_edges)
            ret[ixs] = zero(Real)
            ret
        end
        pos_edges::Matrix{Real} = zeroOut(original_edges .< 0)
        neg_edges::Matrix{Real} = abs(zeroOut(original_edges .> 0))
        Dict(:original => original_edges, :positive => pos_edges, :negative => neg_edges)
     end
         
     originalToConvenince(lr_edges), originalToConvenince(svr_edges)
end

typealias EdgesMap Dict{Symbol, Matrix{Real}}
function saveData(lr_nodes::DataFrame, svr_nodes::DataFrame, 
                  lr_edges::EdgesMap, svr_edges::EdgesMap;
                  dest_dir::ASCIIString="../data/step4/surfice_input")
     writeNodes(f::AbstractString, nodes::DataFrame) = writetable(joinpath(dest_dir, f), 
                                                                  nodes, separator='\t', header=false)
     
     writeNodes("lr.node", lr_nodes)
     writeNodes("svr.node", svr_nodes)
     
     function writeEdges(file_prefix::ASCIIString, edges::EdgesMap)
        for edgeType::Symbol in keys(edges)
            mat::Matrix{Real} = edges[edgeType]
            f::AbstractString = joinpath(dest_dir, "$(file_prefix)_$edgeType.edge")
            writedlm(f, mat, '\t', header=false)
        end
     end
     
     writeEdges("lr", lr_edges)
     writeEdges("svr", svr_edges)
end


function main(lr_lesion_f::AbstractString="../data/step4/linear_reg/adw_all_subjects_left_select_diff_wpm_lesion.csv", 
              svr_lesion_f::AbstractString="../data/step4/svr/adw_all_subjects_left_select_diff_wpm_lesion_predictors.csv",
              lr_conn_f::AbstractString="../data/step4/linear_reg/adw_all_subjects_left_select_diff_wpm_conn.csv", 
              svr_conn_f::AbstractString="../data/step4/svr/adw_all_subjects_left_select_diff_wpm_conn_predictors.csv")
     lr_nodes::DataFrame, svr_nodes::DataFrame = convertLesionsToNodes(lr_lesion_f, svr_lesion_f)
     
     lr_edges::EdgesMap , svr_edges::EdgesMap = convertConnectionsToEdges(lr_conn_f, svr_conn_f,
                                                                          lr_nodes, svr_nodes) 
                                                                          
     saveData(lr_nodes, svr_nodes, lr_edges, svr_edges)
end
