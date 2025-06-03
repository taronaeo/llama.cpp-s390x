if (GGML_ZDNN)
    if (DEFINED ZDNN_ROOT)
        message(STATUS "zdnn: using ZDNN_ROOT override: ${ZDNN_ROOT}")
        set(ZDNN_HINT "${ZDNN_ROOT}")
    else()
        set(ZDNN_HINT "")
    endif()

    find_path(
        ZDNN_INCLUDE_DIRS
        NAMES zdnn.h
        HINTS "${ZDNN_HINT}/include" /usr/include /usr/local/include)
    if (ZDNN_INCLUDE_DIRS)
        message(STATUS "zdnn: includes found: ${ZDNN_INCLUDE_DIRS}")
    else()
        message(FATAL_ERROR "zdnn: includes not found, please set ZDNN_ROOT to the proper location if necessary")
    endif()

    find_library(
        ZDNN_LIBRARIES
        NAMES zdnn
        HINTS "${ZDNN_HINT}/lib" /usr/lib /usr/lib64 /usr/local/lib)
    if(ZDNN_LIBRARIES)
        message(STATUS "zdnn: libraries found: ${ZDNN_LIBRARIES}")
    else()
        message(FATAL_ERROR "zdnn: libraries not found, please set ZDNN_ROOT to the proper location if necessary")
    endif()

    file(GLOB GGML_SOURCES_ZDNN "*.c" "*.cpp")
    file(GLOB GGML_HEADERS_ZDNN "*.h")

    ggml_add_backend_library(ggml-zdnn ${GGML_HEADERS_ZDNN} ${GGML_SOURCES_ZDNN})
    target_link_libraries(ggml-zdnn PRIVATE ${ZDNN_LIBRARIES})
    target_include_directories(ggml-zdnn PRIVATE ${ZDNN_INCLUDE_DIRS})
    target_link_directories(ggml-zdnn PRIVATE ${ZDNN_LIBRARIES})

    target_compile_definitions(ggml-zdnn PRIVATE GGML_USE_ZDNN)
endif()
