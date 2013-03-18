if(OPENNI_FOUND)
    if(NOT VTK_FOUND)
        set(DEFAULT FALSE)
        set(REASON "VTK was not found.")
    else(NOT VTK_FOUND)
        set(DEFAULT TRUE)
        set(REASON)
        set(VTK_USE_FILE ${VTK_USE_FILE} CACHE INTERNAL "VTK_USE_FILE")
        include (${VTK_USE_FILE})
        include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
    endif(NOT VTK_FOUND)

    FILE(GLOB hdrs "*.h*")
    include_directories(${VTK_INCLUDE_DIRS} ${OPENNI_INCLUDE_DIRS})       

    ## KINECT FUSION
    set(the_target pcl_kinfu2_app)
    set(srcs kinfu_app.cpp)

    source_group("Source Files" FILES ${srcs} )

    PCL_ADD_EXECUTABLE(${the_target} ${SUBSYS_NAME} ${srcs} ${hdrs})
    target_link_libraries(${the_target} pcl_common pcl_io ${OPENNI_LIBRARIES} pcl_visualization pcl_gpu_kinfu2)  
  
    if(OpenCV_FOUND)
        target_link_libraries(${the_target} ${OpenCV_LIBS})
    endif()
    
endif(OPENNI_FOUND)
