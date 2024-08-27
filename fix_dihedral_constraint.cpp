/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/ Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Hesam Arabzadeh (University of Missouri)
------------------------------------------------------------------------- */

#include "fix_dihedral_constraint.h"

#include "lammps.h"
#include "error.h"
#include "domain.h"
#include "force.h"
#include "pair.h"
#include "update.h"
#include "atom.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "neigh_request.h"

#include <vector>
#include <utility>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include "/usr/include/eigen3/Eigen/Dense"


using namespace LAMMPS_NS;
using namespace FixConst;


static constexpr double TOLERANCE = 0.05;

/* ---------------------------------------------------------------------- */

FixDihedralConstraint::FixDihedralConstraint(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg){
  //printf("\n   ***  Initializing FixDihedralConstraint with ID %s   ***\n", this->id);

  if (narg < 11) error->all(FLERR, "Illegal fix dihedral_constraint command: Not enough input arguments.");

  int index = 3; // Start reading after the fix ID and group
  while (index < narg) {
    if (strcmp(arg[index++], "dihedral") != 0)
        error->all(FLERR, "Expected 'dihedral' keyword in fix dihedral_constraint command.");

    if (index + 8 > narg)
        error->all(FLERR, "Incomplete dihedral specification.");

    // Read the atom types for the dihedral chain
    DihedralConfig config;
    config.type1 = atoi(arg[index++]);
    config.type2 = atoi(arg[index++]);
    config.type3 = atoi(arg[index++]);
    config.type4 = atoi(arg[index++]);
    config.cut   = atoi(arg[index++]);
    config.params.k = atof(arg[index++]);
    config.params.sign = atof(arg[index++]);
    config.params.multiplicity = atoi(arg[index++]);

    if (config.params.sign != -1 && config.params.sign != 1)
      error->all(FLERR, "Incorrect sign arg for dihedral coefficients");
//    std::cout << " type1   : " << config.type1                << '\n'     
//	      << " type2   : " << config.type2                << '\n'
//	      << " type3   : " << config.type3                << '\n'
//	      << " type4   : " << config.type4                << '\n'
//	      << " cut     : " << config.cut                  << '\n'
//	      << " k       : " << config.params.k             << '\n'
//	      << " sign    : " << config.params.sign          << '\n'
//	      << " multip  : " << config.params.multiplicity  << "\n\n";
//
    if (config.params.sign == 1) {
      config.params.cos_shift = 1;
      config.params.sin_shift = 0;
    } else {
      config.params.cos_shift = -1;
      config.params.sin_shift = 0;
    }

    // Store the configuration
    dihedralConfigs.push_back(config);
  }

}

/* ---------------------------------------------------------------------- */

FixDihedralConstraint::~FixDihedralConstraint() {
}

/* ---------------------------------------------------------------------- */

int FixDihedralConstraint::setmask() {
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixDihedralConstraint::initial_integrate(int /*vflag */) {

  if (!list) {
      error->all(FLERR, "Neighbor list not available in FixDihedralConstraint");
  }
  //printf("\n   *** FixDihedralConstraint::initial_integrate ***\n");

  double **x = atom->x;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int i, j, ii, jj, k, l, itype, jtype, inum, jnum;
  int *ilist, *jlist, *numneigh, **firstneigh;
  double xtmp, ytmp, ztmp, rsq, cut;
  double delx, dely, delz;
  int newton_pair = force->newton_pair;
  
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;
  inum = list->inum;
  //printf("\n Number of atoms to process: %d\n", inum);

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];

    if (!(mask[i] & groupbit)) continue;   
    jlist = firstneigh[i];
    jnum = numneigh[i];
    itype = type[i];
    //std::cout << "\n ** got into i with type " << itype <<  " ** \n";

    //printf("\n   i: %i,  itype: %i x: %f  y: %f  z: %f \n", i+1, itype, x[i][0],x[i][1],x[i][2]);

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj] & NEIGHMASK;

      if (!(mask[j] & groupbit)) continue;   
      jtype = type[j];
      //std::cout << " * got into j with type " << jtype <<  " * \n";

      delx =  x[i][0] - x[j][0];
      dely =  x[i][1] - x[j][1];
      delz =  x[i][2] - x[j][2];

      //printf("\n   j: %i,  jtype: %i x: %f  y: %f  z: %f \n",j+1, jtype, x[j][0],x[j][1],x[j][2]);
 
      rsq = delx*delx + dely*dely + delz*delz;
      //printf("\n r: %f \n", sqrt(rsq));

      for (const auto& config : dihedralConfigs) {
    	if ((itype == config.type2 && jtype == config.type3)){ 
	  //std::cout << " config.type2: " << config.type2 << "   config.type3:  " << config.type3 << std::endl;
	  // Forward configuration: k-i-j-l
          if (rsq < config.cut * config.cut) {
	    //std::cout << " rsq < config.cut * config.cut \n";
            k = findBondedAtomByType(i, config.type1); // Find 'k' bonded to 'i'
            l = findBondedAtomByType(j, config.type4); // Find 'l' bonded to 'j'
            //printf("\n k: %i, type: %i \n", k+1, config.type1);
            //printf("\n l: %i, type: %i \n", l+1, config.type4);
    	      
            if (k != -1 && l != -1)
             calculateAndApplyDihedralForce(k, i, j, l, config.params);
	  }
	}
       	//else if ((jtype == config.type2 && itype == config.type3)) {
  	//  //std::cout << " config.type3: " << config.type3 << "   config.type2:  " << config.type2 << std::endl;
        //  // Reverse configuration: l-j-i-k
        //  if (rsq < config.cut * config.cut) {
	//    //std::cout << " rsq < config.cut * config.cut \n";
        //    l = findBondedAtomByType(i, config.type4); // Find 'l' bonded to 'j'
        //    k = findBondedAtomByType(j, config.type1); // Find 'k' bonded to 'i' 
        //    //printf("\n k: %i, type: %i \n", k+1, config.type1);
        //    //printf("\n l: %i, type: %i \n", l+1, config.type4);
    	//      
	//    if (k != -1 && l != -1)
        //      calculateAndApplyDihedralForce(l, j, i, k, config.params);    
	//  }
	//}
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixDihedralConstraint::calculateAndApplyDihedralForce(int i1, int i2, int i3, int i4, DihedralParams params) {

  int i, m, n;
  double vb1x, vb1y, vb1z, vb2x, vb2y, vb2z, vb3x, vb3y, vb3z, vb2xm, vb2ym, vb2zm;
  double edihedral, f1[3], f2[3], f3[3], f4[3];
  double ax, ay, az, bx, by, bz, rasq, rbsq, rgsq, rg, rginv, ra2inv, rb2inv, rabinv;
  double df, df1, ddf1, fg, hg, fga, hgb, gaa, gbb;
  double dtfx, dtfy, dtfz, dtgx, dtgy, dtgz, dthx, dthy, dthz;
  double c, s, p, sx2, sy2, sz2;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  bool newton_bond = force->newton_bond;


//  edihedral = 0.0;
//  ev_init(eflag, vflag);

  double **x = atom->x;
  double **f = atom->f;


    // 1st bond

    vb1x = x[i1][0] - x[i2][0];
    vb1y = x[i1][1] - x[i2][1];
    vb1z = x[i1][2] - x[i2][2];

    // 2nd bond

    vb2x = x[i3][0] - x[i2][0];
    vb2y = x[i3][1] - x[i2][1];
    vb2z = x[i3][2] - x[i2][2];

    vb2xm = -vb2x;
    vb2ym = -vb2y;
    vb2zm = -vb2z;

    // 3rd bond

    vb3x = x[i4][0] - x[i3][0];
    vb3y = x[i4][1] - x[i3][1];
    vb3z = x[i4][2] - x[i3][2];

    /* c,s calculation
    
    Calculate normals using cross products */    

    ax = vb1y * vb2zm - vb1z * vb2ym;
    ay = vb1z * vb2xm - vb1x * vb2zm;
    az = vb1x * vb2ym - vb1y * vb2xm;
    bx = vb3y * vb2zm - vb3z * vb2ym;
    by = vb3z * vb2xm - vb3x * vb2zm;
    bz = vb3x * vb2ym - vb3y * vb2xm;

    rasq = ax * ax + ay * ay + az * az;
    rbsq = bx * bx + by * by + bz * bz;
    rgsq = vb2xm * vb2xm + vb2ym * vb2ym + vb2zm * vb2zm;
    rg = sqrt(rgsq);

    rginv = ra2inv = rb2inv = 0.0;
    if (rg > 0) rginv = 1.0 / rg;
    if (rasq > 0) ra2inv = 1.0 / rasq;
    if (rbsq > 0) rb2inv = 1.0 / rbsq;
    rabinv = sqrt(ra2inv * rb2inv);

    c = (ax * bx + ay * by + az * bz) * rabinv;
    s = rg * rabinv * (ax * vb3x + ay * vb3y + az * vb3z);

    
    //std::cout << " LAMMPS cos_phi:\n" << c << std::endl;
    //std::cout << " LAMMPS sin_phi:\n" << s << std::endl;

    // error check

    if (c > 1.0 + TOLERANCE || c < (-1.0 - TOLERANCE)){
    char errmsg[512];
    sprintf(errmsg, "Cosine of dihedral angle out of expected range for atoms %d, %d, %d, %d. Computed value: %f", i1, i2, i3, i4, c);
    error->all(FLERR, errmsg);
    }

    if (c > 1.0) c = 1.0;
    if (c < -1.0) c = -1.0;


    m = params.multiplicity;
    p = 1.0;
    ddf1 = df1 = 0.0;

    for (i = 0; i < m; i++) {
      ddf1 = p * c - df1 * s;
      df1 = p * s + df1 * c;
      p = ddf1;
    }

    p = p * params.cos_shift + df1 * params.sin_shift;
    df1 = df1 * params.cos_shift - ddf1 * params.sin_shift;
    df1 *= -m;
    p += 1.0;

    if (m == 0) {
      p = 1.0 + params.cos_shift;
      df1 = 0.0;
    }

    // if (eflag) edihedral = k[type] * p;

    fg = vb1x * vb2xm + vb1y * vb2ym + vb1z * vb2zm;
    hg = vb3x * vb2xm + vb3y * vb2ym + vb3z * vb2zm;
    fga = fg * ra2inv * rginv;
    hgb = hg * rb2inv * rginv;
    gaa = -ra2inv * rg;
    gbb = rb2inv * rg;

    //std::cout << "fg: "  << fg << std::endl;
    //std::cout << "hg: "  << hg << std::endl;
    //std::cout << "fga: " << fga << std::endl;
    //std::cout << "hgb: " << hgb << std::endl;
    //std::cout << "gaa: " << gaa << std::endl;
    //std::cout << "gbb: " << gbb << std::endl;


    dtfx = gaa * ax;
    dtfy = gaa * ay;
    dtfz = gaa * az;
    dtgx = fga * ax - hgb * bx;
    dtgy = fga * ay - hgb * by;
    dtgz = fga * az - hgb * bz;
    dthx = gbb * bx;
    dthy = gbb * by;
    dthz = gbb * bz;

    //std::cout << " dtfx: " << dtfx << std::endl;
    //std::cout << " dtfy: " << dtfy << std::endl;
    //std::cout << " dtfz: " << dtfz << std::endl;
    //std::cout << " dtgx: " << dtgx << std::endl;
    //std::cout << " dtgy: " << dtgy << std::endl;
    //std::cout << " dtgz: " << dtgz << std::endl;
    //std::cout << " dthx: " << dthx << std::endl;
    //std::cout << " dthy: " << dthy << std::endl;
    //std::cout << " dthz: " << dthz << std::endl;


    //df = -k[type] * df1;
    df = -params.k * df1;

    sx2 = df * dtgx;
    sy2 = df * dtgy;
    sz2 = df * dtgz;

    f1[0] = df * dtfx;
    f1[1] = df * dtfy;
    f1[2] = df * dtfz;

    //std::cout << "f1: " << f1[0] << "   " << f1[1] << "   " << f1[2] << std::endl;

    f2[0] = sx2 - f1[0];
    f2[1] = sy2 - f1[1];
    f2[2] = sz2 - f1[2];

    //std::cout << "f2: " << f2[0] << "   " << f2[1] << "   " << f2[2] << std::endl;

    f4[0] = df * dthx;
    f4[1] = df * dthy;
    f4[2] = df * dthz;

    //std::cout << "f4: " << f4[0] << "   " << f4[1] << "   " << f4[2] << std::endl;

    f3[0] = -sx2 - f4[0];
    f3[1] = -sy2 - f4[1];
    f3[2] = -sz2 - f4[2];

    //std::cout << "f3: " << f3[0] << "   " << f3[1] << "   " << f3[2] << std::endl;
    // apply force to each of 4 atoms

    if (newton_bond || i1 < nlocal) {
      f[i1][0] += f1[0];
      f[i1][1] += f1[1];
      f[i1][2] += f1[2];
    }

    if (newton_bond || i2 < nlocal) {
      f[i2][0] += f2[0];
      f[i2][1] += f2[1];
      f[i2][2] += f2[2];
    }

    if (newton_bond || i3 < nlocal) {
      f[i3][0] += f3[0];
      f[i3][1] += f3[1];
      f[i3][2] += f3[2];
    }

    if (newton_bond || i4 < nlocal) {
      f[i4][0] += f4[0];
      f[i4][1] += f4[1];
      f[i4][2] += f4[2];
    }

//    if (evflag)
//      ev_tally(i1, i2, i3, i4, nlocal, newton_bond, edihedral, f1, f3, f4, vb1x, vb1y, vb1z, vb2x,
//               vb2y, vb2z, vb3x, vb3y, vb3z);
}    

/* ---------------------------------------------------------------------- */

//void FixDihedralConstraint::calculateAndApplyDihedralForce(int i1, int i2, int i3, int i4, DihedralParams params) {
//
//  int i, m, n, p;
//  double rasq, rbsq, rgsq, rg, rginv, ra2inv, rb2inv, rabinv;
//  double df, df1, ddf1, fg, hg, fga, hgb, gaa, gbb;
//  int *type = atom->type;
//  int nlocal = atom->nlocal;
//  bool newton_bond = force->newton_bond;
//  double n1_length, n2_length, vb2_length, cos_phi, sin_phi;
//  double **x = atom->x;
//  double **f = atom->f;
//
//  using namespace Eigen;
//
//
////  edihedral = 0.0;
////  ev_init(eflag, vflag);
//
//    Vector3d x1 = Vector3d(atom->x[i1][0], atom->x[i1][1], atom->x[i1][2]);
//    Vector3d x2 = Vector3d(atom->x[i2][0], atom->x[i2][1], atom->x[i2][2]);
//    Vector3d x3 = Vector3d(atom->x[i3][0], atom->x[i3][1], atom->x[i3][2]);
//    Vector3d x4 = Vector3d(atom->x[i4][0], atom->x[i4][1], atom->x[i4][2]);
//
//    // Calculate bond vectors
//
//    // 1st bond
//    Vector3d vb1 = x1 - x2;
//
//    // 2nd bond
//    Vector3d vb2 = x3 - x2;
//    Vector3d vb2m = -vb2;
//
//    // 3rd bond
//    Vector3d vb3 = x4 - x3;
//
//    //std::cout << " Eigen vb11:\n" << vb11 << std::endl;
//
//
//    // c,s calculation
//    
//    // Calculate normals to the planes formed by the bonds
//    Vector3d n1 = vb1.cross(vb2m);
//    Vector3d n2 = vb3.cross(vb2m);
//
//    n1_length = n1.norm();
//    n2_length = n2.norm();
//    vb2_length = vb2.norm();
//
//    rasq = n1.squaredNorm();  // Squared norm of n1
//    rbsq = n2.squaredNorm();  // Squared norm of n2
//    rgsq = vb2.squaredNorm(); // Squared norm of vb2    
//
//    rginv  = (rgsq > 0) ? 1.0 / sqrt(rgsq) : 0.0;
//    ra2inv = (rasq > 0) ? 1.0 / rasq : 0.0;
//    rb2inv = (rbsq > 0) ? 1.0 / rbsq : 0.0;
//    rabinv = sqrt(ra2inv * rb2inv);  // This uses the inverse square norms to compute the geometric mean  
//
//    // Calculate the dihedral angle
//    cos_phi = n1.dot(n2) / (n1_length * n2_length);
//    sin_phi = (n1.cross(n2)).dot(vb2.normalized()) / (n1_length * n2_length);
//
//    // error check
//
//    if (cos_phi > 1.0 + TOLERANCE || cos_phi < (-1.0 - TOLERANCE)){
//    char errmsg[512];
//    sprintf(errmsg, "Cosine of dihedral angle out of expected range for atoms %d, %d, %d, %d. Computed value: %f", i1, i2, i3, i4, cos_phi);
//    error->all(FLERR, errmsg);
//    }
//
//    // Clamping for numerical stability
//    cos_phi = std::max(-1.0, std::min(1.0, cos_phi));
//
//    //std::cout << " Eigen cos_phi:\n" << cos_phi << std::endl;
//    //std::cout << " Eigen sin_phi:\n" << sin_phi << std::endl;
//
//
//    m = params.multiplicity;
//    p = 1.0;
//    ddf1 = df1 = 0.0;
//
//    for (i = 0; i < m; i++) {
//      ddf1 = p * cos_phi - df1 * sin_phi;
//      df1 = p * sin_phi + df1 * cos_phi;
//      p = ddf1;
//    }
//
//    p = p * params.cos_shift + df1 * params.sin_shift;
//    df1 = df1 * params.cos_shift - ddf1 * params.sin_shift;
//    df1 *= -m;
//    p += 1.0;
//
//    if (m == 0) {
//      p = 1.0 + params.cos_shift;
//      df1 = 0.0;
//    }
//
//    // if (eflag) edihedral = k[type] * p;
//
//    fg  = vb1.dot(vb2m);
//    hg  = vb3.dot(vb2m);
//    fga = fg * ra2inv * rginv;
//    hgb = hg * rb2inv * rginv;
//    gaa = -ra2inv * vb2.norm();
//    gbb = rb2inv * vb2.norm();
//
//    //std::cout << "fg: "  << fg << std::endl;
//    //std::cout << "hg: "  << hg << std::endl;
//    //std::cout << "fga: " << fga << std::endl;
//    //std::cout << "hgb: " << hgb << std::endl;
//    //std::cout << "gaa: " << gaa << std::endl;
//    //std::cout << "gbb: " << gbb << std::endl;
//
//    // Force calculation vectors
//    Vector3d dtf = n1 * gaa;
//    Vector3d dtg = n1 * fga - n2 * hgb;
//    Vector3d dth = n2 * gbb;
//
//    //std::cout << "df1: " << df1 << ", p: " << p << std::endl;
//    //std::cout << "dtf: " << dtf.transpose() << std::endl;
//    //std::cout << "dtg: " << dtg.transpose() << std::endl;
//    //std::cout << "dth: " << dth.transpose() << std::endl;
//    //std::cout << " dtf: " << dtf << std::endl;
//    //std::cout << " dtg: " << dtg << std::endl;
//    //std::cout << " dth: " << dth << std::endl;
//
//    // Calculate the differential forces
//    Vector3d df_vec = -params.k * df1 * dtg;
//    Vector3d f1 = -params.k * df1 * dtf;
//    Vector3d f2 = df_vec - f1;
//    Vector3d f4 = -params.k * df1 * dth;
//    Vector3d f3 = -df_vec - f4;
//
//    // apply force to each of 4 atoms
//
//    //std::cout << "Atom " << i1 << " initial force: " << Vector3d(atom->f[i1][0], atom->f[i1][1], atom->f[i1][2]).transpose() << std::endl;
//    //std::cout << "Atom " << i1 << " additional force: " << f1.transpose() << std::endl;
//
//
//    if (newton_bond || i1 < nlocal) {
//      atom->f[i1][0] += f1[0];
//      atom->f[i1][1] += f1[1];
//      atom->f[i1][2] += f1[2];
//      //std::cout << "   *** f1 ran over ***\n";
//    }
//    //std::cout << "AFTER   f1: " << Vector3d(atom->f[i1][0], atom->f[i1][1], atom->f[i1][2]).transpose() << std::endl;
//    //std::cout << "BEFORE  f2: " << atom->f[i2][0] << "   " << atom->f[i2][1] << "   " << atom->f[i2][2] << std::endl;
//
//    if (newton_bond || i2 < nlocal) {
//      atom->f[i2][0] += f2[0];
//      atom->f[i2][1] += f2[1];
//      atom->f[i2][2] += f2[2];
//     // std::cout << "   *** f2 ran over ***\n";
//    }
//    //std::cout << "AFTER   f2: " << atom->f[i2][0] << "   " << atom->f[i2][1] << "   " << atom->f[i2][2] << std::endl;
//    //std::cout << "BEFORE  f3: " << atom->f[i3][0] << "   " << atom->f[i3][1] << "   " << atom->f[i3][2] << std::endl;
//
//    if (newton_bond || i3 < nlocal) {
//      atom->f[i3][0] += f3[0];
//      atom->f[i3][1] += f3[1];
//      atom->f[i3][2] += f3[2];
//      //std::cout << "   *** f3 ran over ***\n";
//    }
//    //std::cout << "AFTER   f3: " << atom->f[i3][0] << "   " << atom->f[i3][1] << "   " << atom->f[i3][2] << std::endl;
//    //std::cout << "BEFORE  f4: " << atom->f[i4][0] << "   " << atom->f[i4][1] << "   " << atom->f[i4][2] << std::endl;
//
//    if (newton_bond || i4 < nlocal) {
//      atom->f[i4][0] += f4[0];
//      atom->f[i4][1] += f4[1];
//      atom->f[i4][2] += f4[2];
//      //std::cout << "   *** f4 ran over ***\n";
//    }
//    //std::cout << "AFTER   f4: " << atom->f[i4][0] << "   " << atom->f[i4][1] << "   " << atom->f[i4][2] << std::endl;
// 
////    if (evflag)
////      ev_tally(i1, i2, i3, i4, nlocal, newton_bond, edihedral, f1, f3, f4, vb1x, vb1y, vb1z, vb2x,
////               vb2y, vb2z, vb3x, vb3y, vb3z);
//} 


int FixDihedralConstraint::findBondedAtomByType(int atomIndex, int targetType) {
    int num_bonded = atom->num_bond[atomIndex];
    //printf(" num_bonded: %i \n", num_bonded);
    int *bonded_atoms = atom->bond_atom[atomIndex];
    //printf(" bonded_atom: ");
    //for (int i=0;i<num_bonded;++i)
    //  printf(" %i ", bonded_atoms[i]);
    //  printf("\n"); 

    int *bonded_types = atom->type; // Assuming you have access to types like this

    //std::cout << "Atom Index: " << atomIndex << ", Num Bonded: " << num_bonded << std::endl;
    for (int b = 0; b < num_bonded; b++) {
        int bonded_atom_index = bonded_atoms[b];
        if (bonded_types[bonded_atom_index] == targetType) {
          //std::cout << "Bonded Atom Index: " << bonded_atom_index << ", Bonded Atom Type: " << bonded_types[bonded_atom_index] << std::endl;
          return bonded_atom_index;
        }
    }
    return -1; // Return -1 if no matching type is found
}

/* ---------------------------------------------------------------------- */

bool FixDihedralConstraint::isEligibleForDihedral(int itype, int jtype) {
    for (const auto& config : dihedralConfigs) {
        if ((config.type1 == itype && config.type2 == jtype) ||
            (config.type1 == jtype && config.type2 == itype)) {
            return true;
        }
    }
    return false;
}


/* ---------------------------------------------------------------------- */
void FixDihedralConstraint::init() {
  double mycutoff = 15.0;
  auto request = neighbor->add_request(this, NeighConst::REQ_FULL);
  request->set_cutoff(mycutoff);
}

/* ---------------------------------------------------------------------- */
void FixDihedralConstraint::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

//double calculateDihedralAngle(double *a, double *b, double *c, double *d) {
//    double v1[3], v2[3], v3[3];
//    double n1[3], n2[3], n1xn2[3];
//
//    // Compute vectors
//    for (int i = 0; i < 3; ++i) {
//        v1[i] = b[i] - a[i];
//        v2[i] = c[i] - b[i];
//        v3[i] = d[i] - c[i];
//    }
//
//    // Calculate normals
//    cross_product(v1, v2, n1);
//    cross_product(v2, v3, n2);
//    cross_product(n1, n2, n1xn2);
//
//    // Dot and cross product magnitudes
//    double x = dot_product(n1, n2);
//    double y = dot_product(n1xn2, v2) / vector_norm(v2);
//
//    // Calculate and return the angle in radians
//    return atan2(y, x);
//}
//
//}
