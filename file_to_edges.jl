function process_btc_file(filename)
    # soc-sign-bitcoinotc.csv
    data = Tuple{Int32, Int32, Float32}[]
    open(filename, "r") do file
        for line in eachline(file) # 1, 2, 3, 1922867398
            parts = split(strip(line), ",")
            if length(parts) == 4
                push!(data, (parse(Int32, parts[1]), 
                            parse(Int32, parts[2]), 
                            parse(Int32, parts[3]) + 11))
            end
        end
    end
    return data
end

function process_usa_road_file(filename)
    # USA-road-d.E.gr
    data = Tuple{Int32, Int32, Float32}[]
    open(filename, "r") do file
        for line in eachline(file)
            parts = split(strip(line))
            if length(parts) == 4 && parts[1] == "a"  # Ensure correct format and first letter is "a"
                push!(data, (parse(Int32, parts[2]), 
                            parse(Int32, parts[3]), 
                            parse(Int32, parts[4])))
            end
        end
    end
    return data
end

