# Verify a fitting count beyond the all-pairs cap and the existing PLY contract.
execute_process(COMMAND "${PROGRAM}" --backend cuda --neighbor-search grid
    --particles 10000 --spacing 0.0125 --dt 0.0001 --steps 3 --dump-every 2 --output "${OUTPUT_DIR}"
    RESULT_VARIABLE status OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT status STREQUAL "0" OR NOT out MATCHES "neighbor_search=grid" OR NOT out MATCHES "Completed 3 steps")
    message(FATAL_ERROR "Grid CLI failed (${status}): ${out} ${err}")
endif()
foreach(frame IN ITEMS 000000 000002 000003)
    set(path "${OUTPUT_DIR}/frame_${frame}.ply")
    if(NOT EXISTS "${path}")
        message(FATAL_ERROR "Missing grid snapshot ${path}")
    endif()
    file(SIZE "${path}" size)
    file(READ "${path}" header LIMIT 119)
    if(NOT size EQUAL 120119 OR NOT header MATCHES "element vertex 10000")
        message(FATAL_ERROR "Unexpected grid PLY size/header: ${path} ${size}")
    endif()
endforeach()
execute_process(COMMAND "${PROGRAM}" --backend cuda --neighbor-search all-pairs --particles 128 --steps 2
    RESULT_VARIABLE status OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT status STREQUAL "0" OR NOT out MATCHES "neighbor_search=all-pairs")
    message(FATAL_ERROR "M2 CLI path failed (${status}): ${out} ${err}")
endif()
