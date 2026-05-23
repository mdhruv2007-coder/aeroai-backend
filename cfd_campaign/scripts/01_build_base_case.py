"""
01_build_base_case.py
AERO.AI MVP — CFD Campaign
Builds the OpenFOAM 10 base case directory (2D RANS, simpleFoam, k-omega SST)
All 50 LHS cases will be cloned from this base case.

Save to:
  /mnt/d/AEROAI/MVP1/cfd_campaign/scripts/
Run with:
  python3 01_build_base_case.py
"""

import os

BASE      = "/mnt/d/AEROAI/MVP1/cfd_campaign"
BASE_CASE = os.path.join(BASE, "base_case")

def write(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    print(f"  wrote: {path.replace(BASE_CASE, 'base_case')}")

# ══════════════════════════════════════════════════════════════════════════════
# system/controlDict
# ══════════════════════════════════════════════════════════════════════════════
write(f"{BASE_CASE}/system/controlDict", """\
FoamFile { version 2.0; format ascii; class dictionary; location "system"; object controlDict; }

application     simpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         2000;
deltaT          1;
writeControl    timeStep;
writeInterval   500;
purgeWrite      2;
writeFormat     ascii;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;

functions
{
    forceCoeffs
    {
        type            forceCoeffs;
        libs            ("libforces.so");
        writeControl    timeStep;
        writeInterval   10;
        log             true;
        patches         (airfoil);
        rho             rhoInf;
        rhoInf          1.225;
        liftDir         (0 1 0);
        dragDir         (1 0 0);
        pitchAxis       (0 0 1);
        magUInf         1.0;       // placeholder — overridden per case
        lRef            1.0;
        Aref            1.0;
        CofR            (0.25 0 0);
    }
}
""")

# ══════════════════════════════════════════════════════════════════════════════
# system/fvSchemes
# ══════════════════════════════════════════════════════════════════════════════
write(f"{BASE_CASE}/system/fvSchemes", """\
FoamFile { version 2.0; format ascii; class dictionary; location "system"; object fvSchemes; }

ddtSchemes      { default steadyState; }
gradSchemes     { default Gauss linear; grad(U) cellLimited Gauss linear 1; }
divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss linearUpwindV grad(U);
    div(phi,k)      bounded Gauss upwind;
    div(phi,omega)  bounded Gauss upwind;
    div((nuEff*dev(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes   { default corrected; }
""")

# ══════════════════════════════════════════════════════════════════════════════
# system/fvSolution
# ══════════════════════════════════════════════════════════════════════════════
write(f"{BASE_CASE}/system/fvSolution", """\
FoamFile { version 2.0; format ascii; class dictionary; location "system"; object fvSolution; }

solvers
{
    p
    {
        solver          GAMG;
        tolerance       1e-7;
        relTol          0.01;
        smoother        GaussSeidel;
    }
    U
    {
        solver          smoothSolver;
        smoother        GaussSeidel;
        tolerance       1e-8;
        relTol          0.1;
    }
    k
    {
        solver          smoothSolver;
        smoother        GaussSeidel;
        tolerance       1e-8;
        relTol          0.1;
    }
    omega
    {
        solver          smoothSolver;
        smoother        GaussSeidel;
        tolerance       1e-8;
        relTol          0.1;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors 1;
    consistent      yes;
    residualControl
    {
        p               1e-5;
        U               1e-5;
        k               1e-5;
        omega           1e-5;
    }
}

relaxationFactors
{
    equations
    {
        U       0.9;
        k       0.7;
        omega   0.7;
    }
}
""")

# ══════════════════════════════════════════════════════════════════════════════
# system/blockMeshDict  — C-mesh topology around unit chord airfoil
# ══════════════════════════════════════════════════════════════════════════════
write(f"{BASE_CASE}/system/blockMeshDict", """\
FoamFile { version 2.0; format ascii; class dictionary; location "system"; object blockMeshDict; }

scale 1;

// C-mesh around unit chord NACA airfoil
// Domain: 20c upstream, 30c downstream, 20c top/bottom
// Airfoil geometry injected per-case via snappyHexMesh / foamyHexMesh
// For 2D: single cell in z, front/back = empty

vertices
(
    (-20  -20  -0.05)   // 0
    ( 30  -20  -0.05)   // 1
    ( 30   20  -0.05)   // 2
    (-20   20  -0.05)   // 3
    (-20  -20   0.05)   // 4
    ( 30  -20   0.05)   // 5
    ( 30   20   0.05)   // 6
    (-20   20   0.05)   // 7
);

blocks
(
    hex (0 1 2 3 4 5 6 7) (200 150 1) simpleGrading (1 1 1)
);

boundary
(
    inlet
    {
        type patch;
        faces ((0 4 7 3) (0 3 2 1));
    }
    outlet
    {
        type patch;
        faces ((1 2 6 5));
    }
    top
    {
        type symmetry;
        faces ((3 7 6 2));
    }
    bottom
    {
        type symmetry;
        faces ((0 1 5 4));
    }
    front
    {
        type empty;
        faces ((4 5 6 7));
    }
    back
    {
        type empty;
        faces ((0 3 2 1));
    }
);
""")

# ══════════════════════════════════════════════════════════════════════════════
# 0/ boundary conditions — placeholders, overridden per-case
# ══════════════════════════════════════════════════════════════════════════════
write(f"{BASE_CASE}/0/U", """\
FoamFile { version 2.0; format ascii; class volVectorField; location "0"; object U; }

dimensions      [0 1 -1 0 0 0 0];
internalField   uniform (1 0 0);   // placeholder

boundaryField
{
    inlet       { type freestream; freestreamValue uniform (1 0 0); }
    outlet      { type freestream; freestreamValue uniform (1 0 0); }
    top         { type symmetry; }
    bottom      { type symmetry; }
    airfoil     { type noSlip; }
    front       { type empty; }
    back        { type empty; }
}
""")

write(f"{BASE_CASE}/0/p", """\
FoamFile { version 2.0; format ascii; class volScalarField; location "0"; object p; }

dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0;

boundaryField
{
    inlet       { type freestreamPressure; freestreamValue uniform 0; }
    outlet      { type freestreamPressure; freestreamValue uniform 0; }
    top         { type symmetry; }
    bottom      { type symmetry; }
    airfoil     { type zeroGradient; }
    front       { type empty; }
    back        { type empty; }
}
""")

write(f"{BASE_CASE}/0/k", """\
FoamFile { version 2.0; format ascii; class volScalarField; location "0"; object k; }

dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 1.5e-5;    // placeholder: 0.1% TI at U=1

boundaryField
{
    inlet       { type freestream; freestreamValue uniform 1.5e-5; }
    outlet      { type freestream; freestreamValue uniform 1.5e-5; }
    top         { type symmetry; }
    bottom      { type symmetry; }
    airfoil     { type kqRWallFunction; value uniform 1.5e-5; }
    front       { type empty; }
    back        { type empty; }
}
""")

write(f"{BASE_CASE}/0/omega", """\
FoamFile { version 2.0; format ascii; class volScalarField; location "0"; object omega; }

dimensions      [0 0 -1 0 0 0 0];
internalField   uniform 1.0;       // placeholder

boundaryField
{
    inlet       { type freestream; freestreamValue uniform 1.0; }
    outlet      { type freestream; freestreamValue uniform 1.0; }
    top         { type symmetry; }
    bottom      { type symmetry; }
    airfoil     { type omegaWallFunction; value uniform 1.0; }
    front       { type empty; }
    back        { type empty; }
}
""")

write(f"{BASE_CASE}/0/nut", """\
FoamFile { version 2.0; format ascii; class volScalarField; location "0"; object nut; }

dimensions      [0 2 -1 0 0 0 0];
internalField   uniform 0;

boundaryField
{
    inlet       { type calculated; value uniform 0; }
    outlet      { type calculated; value uniform 0; }
    top         { type symmetry; }
    bottom      { type symmetry; }
    airfoil     { type nutkWallFunction; value uniform 0; }
    front       { type empty; }
    back        { type empty; }
}
""")

# ══════════════════════════════════════════════════════════════════════════════
# constant/turbulenceProperties
# ══════════════════════════════════════════════════════════════════════════════
write(f"{BASE_CASE}/constant/turbulenceProperties", """\
FoamFile { version 2.0; format ascii; class dictionary; location "constant"; object turbulenceProperties; }

simulationType  RAS;
RAS
{
    RASModel    kOmegaSST;
    turbulence  on;
    printCoeffs on;
}
""")

# ══════════════════════════════════════════════════════════════════════════════
# constant/transportProperties  — air at 15°C, nu adjusted per-case via Re
# ══════════════════════════════════════════════════════════════════════════════
write(f"{BASE_CASE}/constant/transportProperties", """\
FoamFile { version 2.0; format ascii; class dictionary; location "constant"; object transportProperties; }

transportModel  Newtonian;
nu              1.5e-05;   // placeholder — overridden per-case by run script
""")

print(f"\n✅ Base case created at: {BASE_CASE}")
print("   Contains: system/ (controlDict, fvSchemes, fvSolution, blockMeshDict)")
print("             0/      (U, p, k, omega, nut)")
print("             constant/ (turbulenceProperties, transportProperties)")
print("\nNext: run 02_setup_cases.py to clone + configure all 50 cases")
