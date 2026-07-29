from pyscf import gto, scf


if __name__ == '__main__':
    atom = '''
    H      1.2194     -0.1652      2.1600
    C      0.6825     -0.0924      1.2087
    C     -0.7075     -0.0352      1.1973
    H     -1.2644     -0.0630      2.1393
    C     -1.3898      0.0572     -0.0114
    H     -2.4836      0.1021     -0.0204
    C     -0.6824      0.0925     -1.2088
    H     -1.2194      0.1652     -2.1599
    C      0.7075      0.0352     -1.1973
    H      1.2641      0.0628     -2.1395
    C      1.3899     -0.0572      0.0114
    H      2.4836     -0.1022      0.0205
    '''
    basis = 'cc-pVTZ'

    mol = gto.M(atom=atom, basis=basis).set(verbose=4)

    ''' `RHF.pari` method combines PARI for K build and normal density fitting for J build.

        Both `auxbasis` and `schwarz_tol` shown here are their default values.
    '''
    auxbasis = 'cc-pVTZ-JKFIT'
    schwarz_tol = 1e-12
    mf_pari = scf.RHF(mol).pari(auxbasis=auxbasis, schwarz_tol=schwarz_tol)
    mf_pari.kernel()

    # Note that DF will be the fastest among the three methods for this small system.
    mf_df = scf.RHF(mol).density_fit()
    mf_df.kernel()

    mf = scf.RHF(mol)
    mf.kernel()

    err_df = (mf_df.e_tot - mf.e_tot) / mol.natm
    err_pari = (mf_pari.e_tot - mf.e_tot) / mol.natm
    print('')
    print('DF   error: %.6f Ha/atom' % (err_df))
    print('PARI error: %.6f Ha/atom' % (err_pari))
