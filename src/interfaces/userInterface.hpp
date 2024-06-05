/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

/** \file   userInterface.hpp
 \brief  Contains the user interface to MrHyDE.
 \author Created by T. Wildey
 */

#ifndef MRHYDE_USERINTERFACE_H
#define MRHYDE_USERINTERFACE_H

#include "trilinos.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#include "Teuchos_YamlParameterListCoreHelpers.hpp"

#include "preferences.hpp"

#if defined(MrHyDE_ENABLE_MIRAGE)
#include "MirageTranslator.hpp"
#endif

namespace MrHyDE {
  //////////////////////////////////////////////////////////////////////////////////////////////
  // Standard constructor
  //////////////////////////////////////////////////////////////////////////////////////////////
  namespace UserInterfaceFactory {
    Teuchos::RCP<Teuchos::ParameterList> UserInterface(const std::string & filename);
  }
}
#endif
