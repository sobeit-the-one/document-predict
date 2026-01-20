# libpredict - Self-contained document prediction library

# Create shared library
add_library(libpredict SHARED
    src/predict.cpp
)

# Set DLL export definition for Windows
if(WIN32)
    target_compile_definitions(libpredict PRIVATE DOCUMENT_PREDICT_EXPORTS)
endif()

# Add include directories
# PUBLIC so that consumers can find document_predict.h in the root include/
target_include_directories(libpredict PUBLIC
    include  # Root include directory for document_predict.h
    ${CMAKE_SOURCE_DIR}/llama.cpp/common
    ${CMAKE_SOURCE_DIR}/llama.cpp/include
)

# Link libraries
target_link_libraries(libpredict PRIVATE 
    common
    llama 
    ${CMAKE_THREAD_LIBS_INIT}
)

# Set C++ standard
target_compile_features(libpredict PRIVATE cxx_std_20)
