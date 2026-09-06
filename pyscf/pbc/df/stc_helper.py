import numpy as np
from pyscf.pbc import tools
from pyscf import lib
from pyscf.lib import logger

def get_coulG(cell, k=np.zeros(3), exx=False, mf=None, mesh=None, Gv=None,
              wrap_around=True, omega_stc=None, omega = None, **kwargs):

    '''
        omega_dot_Rc: for smoothed truncated couloumb.
        omega: for rsh. Default to be None
    '''

    assert( isinstance(exx, str) )
    assert( exx.lower() in ['vcut_sph', 'vcut_ws'] )
    assert( omega_stc is not None )


    # smooth modification with omega_stc
    if mesh is None:
        mesh = cell.mesh
    if 'gs' in kwargs:
        warnings.warn('cell.gs is deprecated.  It is replaced by cell.mesh,'
                      'the number of PWs (=2*gs+1) along each direction.')
        mesh = [2*n+1 for n in kwargs['gs']]
    if Gv is None:
        Gv = cell.get_Gv(mesh)

    if abs(k).sum() > 1e-9:
        if wrap_around:
            # Here we 'wrap around' the high frequency k+G vectors into their lower
            # frequency counterparts.  Important if you want the gamma point and k-point
            # answers to agree
            from pyscf.pbc.tools.pbc import _Gv_wrap_around
            kG = _Gv_wrap_around(cell, Gv, k, mesh)
        else:
            kG = k + Gv
    else:
        kG = Gv

    absG2 = np.einsum('gi,gi->g', kG, kG)

    if hasattr(mf, 'kpts'):
        kpts = mf.kpts
    else:
        kpts = k.reshape(1,3)
    Nk = len(kpts)

    if omega is None:
        _omega = cell.omega
    else:
        _omega = omega

    # calculate vcut coulG without omega_stc
    if abs(_omega) < 1e-10:
        coulG = tools.get_coulG(cell, k, exx, mf, mesh, Gv, wrap_around, 0.0, **kwargs)
    else:
        # the lr SPH/ lr WS has to be computed
        assert ( (_omega > 0) )
        if not getattr(mf, '_ws_lr_exx', None):
            mf._ws_lr_exx = tools.precompute_lr_exx(cell, kpts, omega = _omega, omega_stc = omega_stc)

        # rebuild if a new omega is specified.
        if abs(mf._ws_lr_exx['alpha'] - _omega) > 1e-9:
            mf._ws_lr_exx = tools.precompute_lr_exx(cell, kpts, omega = _omega, omega_stc = omega_stc)

        coulG = get_truncated_lr_coulG(cell, mf, kpts, exx, kG, absG2, _omega)

    f = np.exp(-absG2*0.25/(omega_stc)**2.)

    v0 = coulG[absG2==0]
    coulG *= f

    if abs(_omega) < 1e-10:
        with np.errstate(divide='ignore',invalid='ignore'):
            coulG += 4*np.pi/absG2 * (1. - f)
    else:
        with np.errstate(divide='ignore',invalid='ignore'):
            coulG += 4*np.pi*(np.exp(-absG2*0.25/(_omega)**2.))/absG2 * (1. - f)

    coulG[absG2==0] = v0 + np.pi/(omega_stc)**2.

    return coulG


def get_truncated_lr_coulG(cell, mf, kpts, exx, kG, absG2, omega):

    assert( isinstance(omega, float) )

    if exx.lower() == 'vcut_sph':
        raise NotImplementedError

    elif exx.lower() == 'vcut_ws':


        kcell = mf._ws_lr_exx['kcell']
        vq = mf._ws_lr_exx['vq']
        vR = mf._ws_lr_exx['vR']
        r_mic = mf._ws_lr_exx['r_mic']
        cache = mf._ws_lr_exx['vq_cache']


        with np.errstate(divide='ignore',invalid='ignore'):
            coulG = 0.0 * 4*np.pi/absG2*(1.0 - np.exp(-absG2/(4*omega**2)))
        coulG[absG2==0] = 0.0
        
        gxyz = np.dot(kG, kcell.lattice_vectors().T)/(2*np.pi)
        shift = (gxyz[0] + .5) % 1 - .5
        gxyz_int = np.rint(gxyz - shift).astype(int)
        if abs(gxyz - gxyz_int - shift).max() > 1e-6:
            raise RuntimeError('k+G vectors are incompatible with the FFT mesh')

        no_shift = abs(shift).max() < 1e-9
        if no_shift:
            exx_vq = vq
        else:
            key = tuple(np.round(shift, 12))
            if key not in cache:
                ''' Note: A grid point on the WS boundary can have multiple degenerate r_mic.
                    The current implementation in `precompute_exx` selects only one of them
                    deterministically. These boundary points have zero measure in the continuous
                    integral, so their contribution vanishes as the FFT mesh is refined. Future
                    implementation may want to collect all degenerate r_mic's and average their
                    phases (i.e., similar to how Wannier interpolation handles boundary images).
                '''
                delta = np.dot(shift, kcell.reciprocal_vectors())
                phase = np.exp(-1j * np.dot(r_mic, delta))
                vG = (kcell.vol / len(phase)) * tools.fftk(
                    vR, kcell.mesh, phase)
                cache[key] = vG.real.copy()
            exx_vq = cache[key]

        mesh = np.asarray(kcell.mesh)
        gxyz = (gxyz_int + mesh)%mesh
        qidx = (gxyz[:,0]*mesh[1] + gxyz[:,1])*mesh[2] + gxyz[:,2]
        lower = -(mesh // 2)
        upper = (mesh - 1) // 2
        is_lt_maxqv = ((gxyz_int >= lower) &
                       (gxyz_int <= upper)).all(axis=1)
        coulG = coulG.astype(exx_vq.dtype)
        coulG[is_lt_maxqv] += exx_vq[qidx[is_lt_maxqv]]


        return coulG
        
    else:
        raise NotImplementedError

    return
