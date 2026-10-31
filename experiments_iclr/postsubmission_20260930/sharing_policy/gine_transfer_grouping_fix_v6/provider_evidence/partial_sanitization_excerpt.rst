Explicit Valence Error - Partial Sanitization
==============================================

| **Author:** Greg Landrum
| **Original Source:** `<https://sourceforge.net/p/rdkit/mailman/message/32599798/>`_
| **Index ID#:** RDKitCB_15
| **Summary:** Create a mol object with skipping valence check, followed by a partial sanitization. N.B. Use caution, and make sure your molecules actually make sense before doing this!

.. testcode::

   from rdkit import Chem
   from rdkit.Chem import rdqueries

The default RDKit behavior is to reject hypervalent P, so you need to set `sanitize=False`:

.. testcode::

   m = Chem.MolFromSmiles('F[P-](F)(F)(F)(F)F.CN(C)C(F)=[N+](C)C',sanitize=False)
   m

.. image:: images/RDKitCB_15_im0.png

The arrangement of the six F around the P is not the octahedral arrangement we would expect because the RDKit has not assigned a hybridization to the P (or any other atoms):

.. testcode::

   # Build a query for the P
   q = rdqueries.AtomNumEqualsQueryAtom(15)

   # Select the first and only P
   phosphorus = m.GetAtomsMatchingQuery(q)[0]

   print(phosphorus.GetHybridization())

.. testoutput::

   UNSPECIFIED

Next, you probably want to at least do a partial sanitization so that the molecule is actually useful.
In this case, setting the hybridization is key:

.. testcode::

   # Regenerate computed properties like implicit valence and ring information
   m.UpdatePropertyCache(strict=False)

   # Apply several sanitization rules
   Chem.SanitizeMol(m,Chem.SanitizeFlags.SANITIZE_FINDRADICALS|Chem.SanitizeFlags.SANITIZE_KEKULIZE|Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|Chem.SanitizeFlags.SANITIZE_SYMMRINGS,catchErrors=True)
   m

.. image:: images/RDKitCB_15_im1.png

Now the expected octahedral arrangement of the six F around the P exists because the hybridization of P has been assigned as SP3D2:

.. testcode::

   print(phosphorus.GetHybridization())

.. testoutput::

   SP3D2


Detect Chemistry Problems
==========================================================

| **Author:** Greg Landrum
| **Source:** `<https://sourceforge.net/p/rdkit/mailman/message/36779572/>`_
| **Index ID#:** RDKitCB_14
| **Summary:** Identify and capture error messages when creating mol objects.

.. testcode::

   from rdkit import Chem
   m = Chem.MolFromSmiles('CN(C)(C)C', sanitize=False)
   problems = Chem.DetectChemistryProblems(m)
   print(len(problems))

.. testoutput::

   1

.. testcode::

   print(problems[0].GetType())
   print(problems[0].GetAtomIdx())
   print(problems[0].Message())

.. testoutput::

   AtomValenceException
   1
   Explicit valence for atom # 1 N, 4, is greater than permitted

.. testcode:: 

   m2 = Chem.MolFromSmiles('c1cncc1',sanitize=False)
   problems = Chem.DetectChemistryProblems(m2)
   print(len(problems))

.. testoutput::

   1

.. testcode::

   print(problems[0].GetType())
   print(problems[0].GetAtomIndices())
   print(problems[0].Message())
   
.. testoutput::
   :options: -ELLIPSIS, +NORMALIZE_WHITESPACE

   KekulizeException
   (0, 1, 2, 3, 4)
   Can't kekulize mol.  Unkekulized atoms: 0 1 2 3 4

Miscellaneous Topics
**********************

Explicit Valence and Number of Hydrogens
==============================================

| **Author:** Michael Palmer and Greg Landrum
| **Source:** `<https://sourceforge.net/p/rdkit/mailman/message/29679834/>`_
| **Index ID#:** RDKitCB_11
| **Summary:** Calculate the explicit valence, number of explicit and implicit hydrogens, and total number of hydrogens on an atom. See the link for an important explanation about terminology and implementation of these methods in RDKit. Highlights are presented below.

Most of the time (exception is explained below), explicit refers to atoms that are in the graph and 
implicit refers to atoms that are not in the graph (i.e., Hydrogens). So given that the ring is aromatic (e.g.,in pyrrole), 
the explicit valence of each of the atoms (ignoring the Hs that are not present in the graph) in pyrrole is 3. If you want the Hydrogen count,
use GetTotalNumHs(); the total number of Hs for each atom is one:

.. testcode::

    from rdkit import Chem
    pyrrole = Chem.MolFromSmiles('C1=CNC=C1')
    for atom in pyrrole.GetAtoms():
       print(atom.GetSymbol(), atom.GetExplicitValence(), atom.GetTotalNumHs())

.. testoutput::

   C 3 1
   C 3 1
   N 3 1
   C 3 1
   C 3 1

In RDKit, there is overlapping nomenclature around the use of the words
"explicit" and "implicit" when it comes to Hydrogens. When you specify the Hydrogens for an atom inside of square brackets 
in the SMILES, it becomes an "explicit" hydrogen as far as atom.GetNumExplicitHs() is concerned. Here is an example:

.. testcode::

   pyrrole = Chem.MolFromSmiles('C1=CNC=C1')
   mol1 = Chem.MolFromSmiles('C1=CNCC1')
   mol2 = Chem.MolFromSmiles('C1=C[NH]CC1')

.. testcode::

   for atom in pyrrole.GetAtoms():
       print(atom.GetSymbol(), atom.GetExplicitValence(), atom.GetNumImplicitHs(), atom.GetNumExplicitHs(), atom.GetTotalNumHs())

.. testoutput::

   C 3 1 0 1
   C 3 1 0 1
   N 3 0 1 1
   C 3 1 0 1
   C 3 1 0 1

.. testcode::
   
    for atom in mol1.GetAtoms():
       print(atom.GetSymbol(), atom.GetExplicitValence(), atom.GetNumImplicitHs(), atom.GetNumExplicitHs(), atom.GetTotalNumHs())

.. testoutput::

   C 3 1 0 1
   C 3 1 0 1
   N 2 1 0 1
   C 2 2 0 2
   C 2 2 0 2

.. testcode::
    
    for atom in mol2.GetAtoms():
       print(atom.GetSymbol(), atom.GetExplicitValence(), atom.GetNumImplicitHs(), atom.GetNumExplicitHs(), atom.GetTotalNumHs())

.. testoutput::

   C 3 1 0 1
   C 3 1 0 1
   N 3 0 1 1
   C 2 2 0 2
   C 2 2 0 2

Wiener Index
=============

| **Author:** Greg Landrum
| **Source:** `<https://sourceforge.net/p/rdkit/mailman/message/36802142/>`_
| **Index ID#:** RDKitCB_12
| **Summary:** Calculate the Wiener index (a topological index of a molecule)

.. testcode::

   from rdkit import Chem
   def wiener_index(m):
       res = 0
       amat = Chem.GetDistanceMatrix(m)
       num_atoms = m.GetNumAtoms()
       for i in range(num_atoms):
           for j in range(i+1,num_atoms):
               res += amat[i][j]
       return res

.. testcode::

   butane = Chem.MolFromSmiles('CCCC')
   print(wiener_index(butane))

.. testoutput::

   10.0

.. testcode::

   isobutane = Chem.MolFromSmiles('CC(C)C')
   print(wiener_index(isobutane))

.. testoutput::

   9.0


Organometallics with Dative Bonds
==================================

| **Author:** Greg Landrum
| **Source:** `<https://sourceforge.net/p/rdkit/mailman/message/36727044/>`_ and `<https://gist.github.com/greglandrum/6cd7aadcdedb1ebcafa9537e8a47e3a4>`_
| **Index ID#:** RDKitCB_19
| **Summary:** Process organometallic SMILES by detecting single bonds between metals and replacing with dative bonds.

.. testcode::

   from rdkit import Chem
   from rdkit.Chem.Draw import IPythonConsole

.. testcode::

    def is_transition_metal(at):
        n = at.GetAtomicNum()
        return (n>=22 and n<=29) or (n>=40 and n<=47) or (n>=72 and n<=79)
    def set_dative_bonds(mol, fromAtoms=(7,8)):
        """ convert some bonds to dative 
    
        Replaces some single bonds between metals and atoms with atomic numbers in fomAtoms
        with dative bonds. The replacement is only done if the atom has "too many" bonds.
    
        Returns the modified molecule.   
    
        """
        pt = Chem.GetPeriodicTable()
        rwmol = Chem.RWMol(mol)
        rwmol.UpdatePropertyCache(strict=False)
        metals = [at for at in rwmol.GetAtoms() if is_transition_metal(at)]
        for metal in metals:
            for nbr in metal.GetNeighbors():
                if nbr.GetAtomicNum() in fromAtoms and \
                   nbr.GetExplicitValence()>pt.GetDefaultValence(nbr.GetAtomicNum()) and \
                   rwmol.GetBondBetweenAtoms(nbr.GetIdx(),metal.GetIdx()).GetBondType() == Chem.BondType.SINGLE:
                    rwmol.RemoveBond(nbr.GetIdx(),metal.GetIdx())
                    rwmol.AddBond(nbr.GetIdx(),metal.GetIdx(),Chem.BondType.DATIVE)
        return rwmol

.. testcode::

   m = Chem.MolFromSmiles('CN(C)(C)[Pt]', sanitize=False)
   m2 = set_dative_bonds(m)
   m2

.. image:: images/RDKitCB_19_im0.png

.. testcode::

   # we can check the bond between nitrogen and platinum
   print(m2.GetBondBetweenAtoms(1,4).GetBondType())

.. testoutput::

   DATIVE

.. testcode::

   # It also shows up in the output SMILES
   # This i