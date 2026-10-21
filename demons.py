import numpy as np, SimpleITK as sitk
def demons_grid(img_src, img_dst, iters=60, sigma=1.5):
    """Diffeomorphic demons on lung-windowed images; returns sampling grid in source coords."""
    f = lambda a: sitk.GetImageFromArray(np.clip((a.astype(np.float32) + 1000) / 1400, 0, 1))
    fix, mov = f(img_dst), f(img_src)
    for im in (fix, mov): im.SetSpacing((5.0, 5.0, 6.0))
    flt = sitk.FastSymmetricForcesDemonsRegistrationFilter()
    flt.SetNumberOfIterations(iters); flt.SetStandardDeviations(sigma)
    disp = flt.Execute(fix, mov)
    D = sitk.GetArrayFromImage(disp)          # (z,y,x,3) in mm, (x,y,z) order
    zz, yy, xx = np.meshgrid(np.arange(48), np.arange(64), np.arange(64), indexing='ij')
    return np.stack([zz + D[..., 2] / 6.0, yy + D[..., 1] / 5.0, xx + D[..., 0] / 5.0]).astype(np.float32)
