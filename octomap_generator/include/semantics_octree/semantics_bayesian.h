#ifndef SEMANTICS_BAYESIAN_H
#define SEMANTICS_BAYESIAN_H

#include <octomap/ColorOcTree.h>
#define NUM_SEMANTICS 3

namespace octomap
{

  /// Structure of semantic color with confidence
  struct ColorWithConfidence
  {
    ColorWithConfidence()
    {
      color = ColorOcTreeNode::Color(255,255,255);
      confidence = 1.;
    }
    ColorWithConfidence(ColorOcTreeNode::Color col, float conf)
    {
      color = col;
      confidence = conf;
    }
    ColorOcTreeNode::Color color;
    float confidence;
    inline bool operator==(const ColorWithConfidence& rhs) const
    {
        return color == rhs.color && confidence == rhs.confidence;
    }
    inline bool operator!=(const ColorWithConfidence& rhs) const
    {
        return color != rhs.color || confidence != rhs.confidence;
    }
    inline bool operator<(const ColorWithConfidence& rhs) const
    {
      return confidence < rhs.confidence;
    }
    inline bool operator>(const ColorWithConfidence& rhs) const
    {
      return confidence > rhs.confidence;
    }
  };

  std::ostream& operator<<(std::ostream& out, ColorWithConfidence const& c);
  /// Structure contains semantic colors and their confidences
  struct SemanticsBayesian
  {
    ColorWithConfidence data[NUM_SEMANTICS]; ///<Semantic colors and confidences, ordered by confidences

    SemanticsBayesian()
    {
      for(int i = 0; i < NUM_SEMANTICS; i++)
      {
        data[i] = ColorWithConfidence();
      }
    }

    bool operator==(const SemanticsBayesian& rhs) const
    {
        for(int i = 0; i < NUM_SEMANTICS; i++)
        {
          if(data[i] != rhs.data[i])
          {
            return false;
            break;
          }
        }
        return true;
    }

    bool operator!=(const SemanticsBayesian& rhs) const
    {
        return !(*this == rhs);
    }

    ColorOcTreeNode::Color getSemanticColor() const
    {
      return data[0].color;
    }

    bool isSemanticsSet() const
    {
      for(int i = 0; i < NUM_SEMANTICS; i++)
      {
        if(data[i].color != ColorOcTreeNode::Color(255,255,255))
          return true;
      }
      return false;
    }

    /// Perform bayesian fusion
    static SemanticsBayesian semanticFusion(const SemanticsBayesian s1, const SemanticsBayesian s2);
  };

  std::ostream& operator<<(std::ostream& out, SemanticsBayesian const& s);
}
#endif //SEMANTICS_BAYESIAN_H
