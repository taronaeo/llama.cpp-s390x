if (GGML_ZDNN)
    file(GLOB   GGML_SOURCES_ZDNN "*.cpp")
    list(APPEND GGML_HEADERS_ZDNN "../../include/ggml-zdnn.h")

    ggml_add_backend_library(ggml-zdnn ${GGML_HEADERS_ZDNN} ${GGML_SOURCES_ZDNN})
    add_compile_definitions(GGML_USE_ZDNN) # Double check if required, may already be done by ggml-cpu

    if (DEFINED ZDNN_ROOT)
        message(STATUS "Using ZDNN_ROOT override: ${ZDNN_ROOT}")
        set(ZDNN_HINT "${ZDNN_ROOT}/include")
    else()
        set(ZDNN_HINT "")
    endif()

    find_path(ZDNN_INCLUDE_DIR
                NAMES zdnn.h
                HINTS ${ZDNN_HINT} /usr/include /usr/local/include
    )
    if (ZDNN_INCLUDE_DIR)
        message(STATUS "zDNN found, Includes: ${ZDNN_INCLUDE_DIR}")
    else()
        message(FATAL_ERROR "zDNN not found, please set ZDNN_ROOT to the proper location if necessary.")
    endif()

    find_library(ZDNN_LIBRARY_DIR
                    NAMES zdnn
                    HINTS ${ZDNN_HINT} /usr/lib /usr/lib64 /usr/local/lib
    )
    if (ZDNN_LIBRARY_DIR)
        message(STATUS "zDNN found, Library: ${ZDNN_LIBRARY_DIR}")
    else()
        message(FATAL_ERROR "zDNN not found, please set ZDNN_ROOT to the proper location if necessary.")
    endif()

    target_link_libraries(ggml-zdnn PRIVATE ${ZDNN_LIBRARY_DIR})
    target_include_directories(ggml-zdnn PRIVATE ${ZDNN_INCLUDE_DIR})

    set(ZDNN_FLAGS -lzdnn)
    target_compile_options(ggml-zdnn PRIVATE ${ZDNN_FLAGS})
endif()

