#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <vector>
#include <string>
#include <algorithm>
#include <cctype>


enum TableState {
    NONE,
    PROTEIN,
    SITE,
    INTERACTING_SITES,
    INTERACTING_PARTICLES
};

struct Protein {
    std::string lammpsType;
    std::string fragName;
    std::string uniProtID;
    double radius;
    double mass;
};

struct Site {
    std::string protID;
    std::string siteID;
    std::vector<double> coords;
};

struct InteractingSites {
    std::string  intID;
    std::vector<double> lockKeyPrm;
}; 

struct InteractingParticles {
    std::string intPID;
    std::vector<double> vdW;
    std::vector<double> Pauli;
    std::vector<double> electrostatic;
};


int main(){

  std::map<std::string, Protein> proteinTable;
  std::map<std::string, std::vector<Site>> siteTable;
  std::map<std::string, InteractingSites> interactingSitesTable;
  std::map<std::string, InteractingParticles> interactingParticlesTable;


  std::ifstream file("table.csv");
  std::string line;
  int line_counter = 0;
  TableState currentState = NONE;

  while (std::getline(file, line)) {

    std::istringstream ss(line);
    std::vector<std::string> cells;
    std::string cell;
    line_counter += 1;
     
    while (std::getline(ss, cell, ',')) {
      cell.erase(remove_if(cell.begin(), cell.end(), isspace), cell.end());
      cells.push_back(cell);
    }
  
    // Determine the current state (which table we are reading)
    if (line == "# Protein table") {
      currentState = PROTEIN;
    } 
    else if (line == "# Site table") {
      currentState = SITE;
    } 
    else if (line == "# Interacting sites table") {
      currentState = INTERACTING_SITES;
    } 
    else if (line == "# Interacting particles table") {
      currentState = INTERACTING_PARTICLES;
    } 
    else if (line.find_first_not_of(" \t\n\r") != std::string::npos && line[0] != '#') {
      // Process the data row
      while (std::getline(ss, cell, ',')) {
          cell.erase(remove_if(cell.begin(), cell.end(), isspace), cell.end());
          cells.push_back(cell);
      }

      // Handle data based on the current state
      switch (currentState) {
          case PROTEIN: {
              if (!(cells.size() ==  5)) {
                std::cerr << "Error: on line " << line_counter << " expected 5 columns, but got " << cells.size() << ", on line: " << line << "\n\n";
                return -1;
              }
              Protein proteinEntry;
              proteinEntry.lammpsType = cells[0];
              proteinEntry.fragName   = cells[1];
              proteinEntry.uniProtID = cells[2];

              try {
              proteinEntry.radius = std::stod(cells[3]);
              }
              catch (const std::invalid_argument& ia) {
                std::cerr << "Invalid argument: " << ia.what() << '\n'
                          << "On line " << line_counter << " could not convert value '" << cells[3] << "' to integer on line: " << line << "\n\n";
                return -1;
              }
              try {
              proteinEntry.mass = std::stod(cells[4]);
              }
              catch (const std::invalid_argument& ia) {
                std::cerr << "Invalid argument: " << ia.what() << '\n'
                          << "On line " << line_counter << " could not convert value '" << cells[4] << "' to double on line: " << line << "\n\n";
                return -1;
              }
              //Insert the protein data into the proteinTable using LAMMPS type as the key
              proteinTable[proteinEntry.lammpsType] = proteinEntry;
              break;     

          } 

          case SITE: {

              if (!(cells.size() ==  5)) {
                std::cerr << "Error: on line " << line_counter << " expected 5 columns, but got " << cells.size() << ", on line: " << line << "\n\n";
                return -1;
              }
              Site  siteEntry;
              siteEntry.protID = cells[0];
              siteEntry.siteID = cells[1];
              siteEntry.coords.resize(3);
              for (int i=0;i<3;++i){
                  try {
                    siteEntry.coords[i] = std::stod(cells[i+2]);
                  }
                  catch (const std::invalid_argument& ia) {
                    std::cerr << "Invalid argument: " << ia.what() << '\n'
                      << "On line " << line_counter << " could not convert value '" << cells[i+2] << "' to double on line: " << line << "\n\n";
                    return -1;
                  }
              }
              // If the protein ID does not exist in the siteTable, add it with an empty vector
              if (siteTable.find(siteEntry.protID) == siteTable.end()) {
                  siteTable[siteEntry.protID] = std::vector<Site>();
              }

              // Append the site to the vector of sites associated with the protein ID
              siteTable[siteEntry.protID].push_back(siteEntry);

              break;
          }

          case INTERACTING_SITES: {

              if (!(cells.size() ==  5)) {
                std::cerr << "Error: on line " << line_counter << " expected 5 columns, but got " << cells.size() << ", on line: " << line << "\n\n";
                return -1;
              }
              InteractingSites interactingSitesEntry;
              interactingSitesEntry.intID = cells[0];
              interactingSitesEntry.lockKeyPrm.resize(4);

              for (int i=0;i<4;++i){
                try{
                  interactingSitesEntry.lockKeyPrm[i] = std::stod(cells[i+1]);
                }
                catch (const std::invalid_argument& ia) {
                  std::cerr << "Invalid argument: " << ia.what() << '\n'
                    << "On line " << line_counter << " could not convert value '" << cells[i+1] << "' to double on line: " << line << "\n\n";
                  return -1;
                }
              }

              // Append the site to the vector of sites associated with the protein ID
              interactingSitesTable[interactingSitesEntry.intID] = interactingSitesEntry;

              break;
          }
          case INTERACTING_PARTICLES: {
              if (!(cells.size() ==  13)) {
                std::cerr << "Error: on line " << line_counter << " expected 13 columns, but got " << cells.size() << ", on line: " << line << "\n\n";
                return -1;
              }
              InteractingParticles interactingParticlesEntry;
              interactingParticlesEntry.intPID = cells[0];
              interactingParticlesEntry.vdW.resize(4);
              interactingParticlesEntry.Pauli.resize(4);
              interactingParticlesEntry.electrostatic.resize(4);
              for (int i=0;i<4;++i){
                try{
                  interactingParticlesEntry.vdW[i] = std::stod(cells[i+1]);
                }
                catch (const std::invalid_argument& ia) {
                  std::cerr << "Invalid argument: " << ia.what() << '\n'
                    << "On line " << line_counter << " could not convert value '" << cells[i+1] << "' to double on line: " << line << "\n\n";
                  return -1;
                }
              }
              for (int i=0;i<4;++i){
                try{
                  interactingParticlesEntry.Pauli[i] = std::stod(cells[i+5]);
                }
                catch (const std::invalid_argument& ia) {
                  std::cerr << "Invalid argument: " << ia.what() << '\n'
                    << "On line " << line_counter << " could not convert value '" << cells[i+5] << "' to double on line: " << line << "\n\n";
                  return -1;
                }
              }
              for (int i=0;i<4;++i){
                try{
                  interactingParticlesEntry.electrostatic[i] = std::stod(cells[i+9]);
                }
                catch (const std::invalid_argument& ia) {
                  std::cerr << "Invalid argument: " << ia.what() << '\n'
                    << "On line " << line_counter << " could not convert value '" << cells[i+9] << "' to double on line: " << line << "\n\n";
                  return -1;
                }
              }

              // Append the site to the vector of sites associated with the protein ID
              interactingParticlesTable[interactingParticlesEntry.intPID] = interactingParticlesEntry;    

              break;
          }
          default: {
              // Handle unrecognized states or lines if necessary
              std::cerr << "Warning: on line " << line_counter << " encountered line outside of recognized tables: " << line << "\n";
              break;
          }
        }
    }
  }

  // Example: let's say we want to get the interaction parameters for all pairs of sites in proteins 1 and 2
  std::cout << "\n";

  std::string particle1_id = "1";
  std::string particle2_id = "3";

  for(int it=0;it<3; ++it){
  // Check if particle1_id exists in the siteTable
  if (siteTable.count(particle1_id) == 0) continue;

  std::cout << "  **** iteration " << it << " is running! **** \n";
//    std::cout << " the particle ID: " << particle1_id << " exists in the site Table! \n";

  std::string lammpsType1 = proteinTable[particle1_id].lammpsType;
  std::string lammpsType2 = proteinTable[particle2_id].lammpsType;

  std::cout << "lammpsType1: " << lammpsType1 << '\n';
  std::cout << "lammpsType2: " << lammpsType2 << '\n';

  int numSites1 = siteTable[particle1_id].size();
  int numSites2 = siteTable[particle2_id].size();

  std::cout << "number of Sites 1: " << numSites1 << '\n';
  std::cout << "number of Sites 2: " << numSites2 << '\n';

  std::string p1site1ID = siteTable[particle1_id][0].siteID;
  std::string p1site2ID = siteTable[particle1_id][1].siteID;
  std::string p2site1ID = siteTable[particle2_id][0].siteID;

  std::cout << "Particle 1 site ID 1: " << p1site1ID << '\n';
  std::cout << "particle 1 site ID 2: " << p1site2ID << '\n';
  std::cout << "particle 2 site ID 1: " << p2site1ID << '\n';
  std::cout << '\n';
  std::cout << '\n';

  for (size_t i = 0; i < siteTable[particle1_id].size(); ++i) {
    for (size_t j = 0; j < siteTable[particle2_id].size(); ++j) {
      std::string site1_id = siteTable[particle1_id][i].siteID;
      std::string site2_id = siteTable[particle2_id][j].siteID;
      std::string interaction_id = (site1_id < site2_id) ? (site1_id + "-" + site2_id) : (site2_id + "-" + site1_id);

      if (interactingSitesTable.find(interaction_id) != interactingSitesTable.end()) {
        // Use foundInteraction for further processing...
        InteractingSites foundInteraction = interactingSitesTable[interaction_id];
        std::cout << "Interaction sites found for  " << interaction_id << "\n\n";
	std::cout << "coordinates of particle 1's site " <<  siteTable[particle1_id][i].siteID << " : ";
	for (int k=0; k<3;++k) {
	  std::cout <<  siteTable[particle1_id][i].coords[k] << "   ";
	}
        std::cout << '\n';
	std::cout << "coordinates of particle 2's site " <<  siteTable[particle2_id][j].siteID << " : ";
	for (int k=0; k<3;++k) {
	  std::cout <<  siteTable[particle2_id][j].coords[k] << "   ";
	}
        std::cout << "\n\n";
	std::cout << "lock and key parameters of particle 1's site " <<  siteTable[particle1_id][i].siteID << " : ";
	for (int k=0; k<4;++k) {
	  std::cout <<  interactingSitesTable[interaction_id].lockKeyPrm[k] << "   ";
	}
        std::cout << '\n';
	std::cout << "lock and key parameters of particle 2's site " <<  siteTable[particle2_id][j].siteID << " : ";
	for (int k=0; k<4;++k) {
	  std::cout <<  interactingSitesTable[interaction_id].lockKeyPrm[k] << "   ";
	}
      }
      else {
        std::cout << "NO interaction found for  " << interaction_id << '\n';
      }
      std::cout << "\n";
    }
  }

  std::cout << "proteinTable.size(): " << proteinTable.size() << '\n';

  std::string p1_id = proteinTable[particle1_id].lammpsType;
  std::string p2_id = proteinTable[particle2_id].lammpsType; 
  std::string interaction_id = (p1_id < p2_id) ? (p1_id + "-" + p2_id) : (p2_id + "-" + p1_id);

  if (interactingParticlesTable.find(interaction_id) != interactingParticlesTable.end()) {

    std::cout << "Interaction sites found for  " << interaction_id << "\n\n";
    std::cout << '\n';
    std::cout << "vdW parameters of particle " <<  p1_id << ": ";
    for (int k=0; k<4;++k) {
      std::cout <<  interactingParticlesTable[interaction_id].vdW[k] << "   ";
    }
    std::cout << '\n';
    std::cout << "Pauli parameters of particle " <<  p1_id << ": ";
    for (int k=0; k<4;++k) {
      std::cout <<  interactingParticlesTable[interaction_id].Pauli[k] << "   ";
    }
    std::cout << '\n';
    std::cout << "Electrostatic parameters of particle " <<  p1_id << ": ";
    for (int k=0; k<4;++k) {
      std::cout <<  interactingParticlesTable[interaction_id].electrostatic[k] << "   ";
    }
    std::cout << '\n';
  }
  }

  file.close();

  return 0;
}
