#ifndef SEMANTICS_POINT_TYPE
#define SEMANTICS_POINT_TYPE
// Reference http://pointclouds.org/documentation/tutorials/adding_custom_ptype.php
#define PCL_NO_PRECOMPILE
#include <pcl/point_types.h>
/**
 * \brief Point type contains XYZ RGB, 3 most confident semantic colors and their confidences
 * \author Xuan Zhang
 * \data Mai-July 2018
 */

struct PointXYZRGBSemanticsMax
{
  PCL_ADD_POINT4D; // Preferred way of adding a XYZ+padding
  PCL_ADD_RGB;
  union  // Semantic color
  {
    float semantic_color;
  };
  union  // Confidences
  {
    float confidence;
  };
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

// here we assume a XYZ + RGB + "sementic_color" + "confidence" (as fields)
POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZRGBSemanticsMax,
                                   (float, x, x)
                                   (float, y, y)
                                   (float, z, z)
                                   (float, rgb, rgb)
                                   (float, semantic_color, semantic_color)
                                   (float, confidence, confidence)
)

struct PointXYZRGBSemanticsBayesian
{
  PCL_ADD_POINT4D;                  // Preferred way of adding a XYZ+padding
  PCL_ADD_RGB;
  union  // Semantic colors
  {
      float data_sem[4];
      struct
      {
        float semantic_color1;
        float semantic_color2;
        float semantic_color3;
      };
  };
  union  // Confidences
  {
    float data_conf[4];
    struct
    {
      float confidence1;
      float confidence2;
      float confidence3;
    };
  };
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

// here we assume a XYZ + RGB + "sementic_colors" + "confidences" (as fields)
POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZRGBSemanticsBayesian,
                                   (float, x, x)
                                   (float, y, y)
                                   (float, z, z)
                                   (float, rgb, rgb)
                                   (float, semantic_color1, semantic_color1)
                                   (float, semantic_color2, semantic_color2)
                                   (float, semantic_color3, semantic_color3)
                                   (float, confidence1, confidence1)
                                   (float, confidence2, confidence2)
                                   (float, confidence3, confidence3)
)
#endif
