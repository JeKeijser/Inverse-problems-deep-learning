import deeptrack as dt
from typing import Tuple, Dict
import numpy as np
from deeptrack.holography import get_propagation_matrix
from deeptrack import image
from deeptrack.backend.units import get_active_voxel_size
from deeptrack import pad_image_to_fft
from deeptrack.types import ArrayLike
import os
import time

np.random.seed(42)

def bimod(r_low, r_high, p):
    r = np.random.rand()
    if r < p:
        RI = r_low + .02 * np.random.randn()
        if RI < 1.34:
            dRI = 1.34 - RI
            RI = 1.34 + dRI
    else:
        RI = r_high + .03 * np.random.randn()
    return RI


class NonSphericalParticle(dt.MieScatterer):

    def _process_properties(self, properties: Dict) -> Dict:
        properties = super()._process_properties(properties)
        return properties

    def __init__(
        self,
        radius: dt.PropertyLike[float] = 1e-6,
        refractive_index: dt.PropertyLike[float] = 1.45,
        alpha: dt.PropertyLike[float] = 1,
        angle: dt.PropertyLike[float] = 0,
        halo: dt.PropertyLike[float] = 0,
        **kwargs,
    ) -> None:

        def coeffs(radius, refractive_index, refractive_index_medium, wavelength):
            if isinstance(radius, dt.Quantity):
                radius = radius.to("m").magnitude
            if isinstance(wavelength, dt.Quantity):
                wavelength = wavelength.to("m").magnitude

            def inner(L):
                return dt.mie.coefficients(
                    refractive_index / refractive_index_medium,
                    radius * 2 * np.pi / wavelength * refractive_index_medium,
                    L,
                )
            return inner

        super().__init__(
            coefficients=coeffs,
            radius=radius,
            refractive_index=refractive_index,
            alpha=alpha,
            halo=halo,
            angle=angle,
            **kwargs,
        )

    def get_plane_in_polar_coords(
        self,
        shape: int,
        voxel_size: ArrayLike,
        plane_position: float,
        illumination_angle: float,
        k: float
    ) -> Tuple[float, float, float, float]:

        X, Y = self.get_XY(shape, voxel_size)
        X = image.maybe_cupy(X)
        Y = image.maybe_cupy(Y)

        X = X + plane_position[0]
        Y = Y + plane_position[1]
        Z = plane_position[2]

        R2_squared = X ** 2 + Y ** 2
        R3 = np.sqrt(R2_squared + Z ** 2)
        Q = np.sqrt(R2_squared)/voxel_size[0]**2*2*np.pi/shape[0]
        sin_theta = 1.52/1.33*Q/(k)

        pupil_mask = sin_theta < 1
        QV = np.zeros(X.shape)

        QV[pupil_mask] = 2*k*np.sin(np.arcsin(sin_theta[pupil_mask]/2))
        cos_theta = np.zeros(sin_theta.shape)
        cos_theta[pupil_mask] = np.sqrt(1-sin_theta[pupil_mask]**2)

        illumination_cos_theta = (
            np.cos(np.arccos(cos_theta) + illumination_angle)
        )
        phi = np.arctan2(Y, X)

        return R3, cos_theta, illumination_cos_theta, phi, pupil_mask, QV

    def get(
        self,
        inp,
        position: ArrayLike,
        voxel_size: ArrayLike,
        padding: ArrayLike,
        wavelength: float,
        refractive_index_medium: float,
        L,
        collection_angle: float,
        input_polarization: float,
        output_polarization: float,
        coefficients,
        offset_z: float,
        z: float,
        working_distance: float,
        position_objective: float,
        coherence_length: float,
        output_region: ArrayLike,
        illumination_angle: float,
        amp_factor: float,
        phase_shift_correction: bool,
        radius: float,
        angle: float,
        halo: float,
        refractive_index: float,
        return_fft=False,
        pupil: ArrayLike = [],
        **kwargs,
    ):
        sigma_x = radius[0]
        sigma_y = radius[1]
        RI = refractive_index

        xSize, ySize = self.get_xy_size(output_region, padding)
        voxel_size = get_active_voxel_size()
        arr = pad_image_to_fft(np.zeros((xSize, ySize))).astype(complex)
        arr = image.maybe_cupy(arr)
        position = np.array(position) * voxel_size[: len(position)]

        pupil_physical_size = working_distance * np.tan(collection_angle) * 2
        z = z * voxel_size[2]
        ratio = offset_z / (working_distance - z)
        k = 2 * np.pi / wavelength * refractive_index_medium

        relative_position = np.array((
            position_objective[0] - position[0],
            position_objective[1] - position[1],
            working_distance - z,
        ))

        R3_field, cos_theta_field, illumination_angle_field, phi_field, pupil_mask, QV = \
            self.get_plane_in_polar_coords(
                arr.shape, voxel_size, relative_position * ratio, illumination_angle, k
            )

        cos_phi_field, sin_phi_field = np.cos(phi_field), np.sin(phi_field)

        R_field_x = sigma_x*QV[pupil_mask].real*cos_phi_field[pupil_mask]
        R_field_y = sigma_y*QV[pupil_mask].real*sin_phi_field[pupil_mask]

        R_field_1 = np.sin(angle)*R_field_x + np.cos(angle)*R_field_y
        R_field_2 = -np.sin(angle)*R_field_y + np.cos(angle)*R_field_x

        R_field = np.sqrt(R_field_1**2 + R_field_2**2)

        form_fac = ((sigma_x*sigma_y)**(3/2)+(halo/2)**3)*4*np.pi/3*k*1/R_field**3*(np.sin(R_field)-R_field*np.cos(R_field))
        form_fac = form_fac/(1+(halo*QV[pupil_mask])**2)

        arr[pupil_mask] = (RI-1.33)*form_fac

        if phase_shift_correction:
            arr *= np.exp(1j * k * z + 1j * np.pi / 2)

        if coherence_length:
            sigma = z * np.sqrt((coherence_length / z + 1) ** 2 - 1)
            sigma = sigma * (offset_z / z)
            mask = np.zeros_like(arr)
            y, x = np.ogrid[
                -mask.shape[0] // 2: mask.shape[0] // 2,
                -mask.shape[1] // 2: mask.shape[1] // 2,
            ]
            mask = np.exp(-0.5 * (x ** 2 + y ** 2) / ((sigma) ** 2))
            mask = image.maybe_cupy(mask)
            arr = arr * mask

        if len(pupil) > 0:
            c_pix = [arr.shape[0]//2, arr.shape[1]//2]
            arr[c_pix[0]-pupil.shape[0]//2:c_pix[0]+pupil.shape[0]//2,
                c_pix[1]-pupil.shape[1]//2:c_pix[1]+pupil.shape[1]//2] *= pupil

        fourier_field = np.fft.ifft2(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(arr))))

        propagation_matrix = get_propagation_matrix(
            fourier_field.shape,
            pixel_size=voxel_size[2],
            wavelength=wavelength / refractive_index_medium,
            to_z=(-z),
            dy=(relative_position[0] * ratio + position[0] + (padding[0] - arr.shape[0] / 2) * voxel_size[0]),
            dx=(relative_position[1] * ratio + position[1] + (padding[1] - arr.shape[1] / 2) * voxel_size[1]),
        )
        fourier_field = fourier_field * propagation_matrix

        if return_fft:
            return -voxel_size[0]**(-2)*fourier_field[..., np.newaxis]
        else:
            return -voxel_size[0]**(-2)*np.fft.ifft2(fourier_field)[..., np.newaxis]
        
def build_pipeline():
    p = NonSphericalParticle(
        radius=lambda: np.random.uniform(.1e-6, .4e-6, size=2),
        refractive_index=lambda: bimod(1.36, 1.45, .4),
        position=lambda: np.random.uniform(25, 38, size=2),
        z=lambda: np.random.uniform(-4, 4),
        halo=lambda: np.random.uniform(0, .5*1e-6),
        angle=lambda: np.random.uniform(0, 2*np.pi)
    )

    def wavynoise(lam, ang, Amp):
        def inner(image):
            x, y = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]))
            return image + Amp * np.sin((x * np.cos(ang) + y * np.sin(ang)) / lam)[..., None]
        return inner

    o = dt.Brightfield(wavelength=430e-9, resolution=80e-9, magnification=1, output_region=(0, 0, 64, 64), return_field=False)
    noise = dt.Gaussian(sigma=lambda: np.random.uniform(.4, 4))
    noise2 = dt.Gaussian(sigma=lambda: np.random.uniform(0.0004, .004))

    s = o(p >> noise) >> noise2 \
        >> dt.Lambda(wavynoise, lam=lambda: np.random.uniform(15, 100), Amp=lambda: np.random.uniform(0.01, .04), ang=lambda: np.random.uniform(0, 2*np.pi)) \
        >> dt.Lambda(wavynoise, lam=lambda: np.random.uniform(2, 10), Amp=lambda: np.random.uniform(0.01, .04), ang=lambda: np.random.uniform(0, 2*np.pi)) \
        & p.radius & p.refractive_index & p.halo

    return s, p

def generate_dataset(n_samples=10000, save_path="data/raw/"):
    
    os.makedirs(save_path, exist_ok=True)
    s, p = build_pipeline()

    X = np.zeros((n_samples, 64, 64))
    y = np.zeros((n_samples, 4))  # [RI, radius_x, radius_y, halo]

    print(f"Generating {n_samples} samples...")
    start = time.time()

    for i in range(n_samples):
        sample = s.update()()
        X[i] = sample[0][..., 0]
        y[i, 0] = sample[2]          # RI
        y[i, 1] = sample[1][0]       # radius_x
        y[i, 2] = sample[1][1]       # radius_y
        y[i, 3] = sample[3]          # halo

        if (i+1) % 1000 == 0:
            print(f"{i+1}/{n_samples} samples, time elapsed: {time.time()-start:.1f}s")

    np.save(os.path.join(save_path, "X.npy"), X)
    np.save(os.path.join(save_path, "y.npy"), y)
    print(f"Done! Saved to {save_path}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")


if __name__ == "__main__":
    generate_dataset(n_samples=10000)