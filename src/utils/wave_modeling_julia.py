import os
import time

import numpy as np


class WaveModeling:
    """wave modeling client working with associated julia code."""

    def __init__(self, path: str, timeout: float = np.inf):
        """

        Args:
            path (str): The folder for read/write commands and data.
            timeout (float): time limit (in seconds) for computation. Default inf.
        """
        self.path = path
        self.timeout = timeout

        os.makedirs(self.path, exist_ok=True)

        self.is_ready = False
        self.last_compute_f: list = []  # [x, f(x)]
        self.last_compute_g: list = []  # [x, g(x)]
        self.last_compute_H: list = []  # [x, v, H(x)v] # we don't use it

    def check_ready(self):
        if self.is_ready:
            return
        path = os.path.join(self.path, "ready.out")
        tic = time.time()
        while not os.path.isfile(path):
            if time.time() - tic > self.timeout + 300:
                raise TimeoutError("Julia wave modeling did not start.")
            time.sleep(1)
        self.is_ready = True

    def compute_f(self, m: np.ndarray) -> float:
        """Compute cost

        Args:
            m (np.ndarray): 1D array velocity field

        Returns:
            float: cost
        """
        # Ensure the julia code is ready
        self.check_ready()

        # Seach history
        if self.last_compute_f:
            if np.allclose(m, self.last_compute_f[0], 1e-15, 1e-15):
                return self.last_compute_f[1]

        # Define paths
        c_path = os.path.join(self.path, "cmd.txt")
        m_path = os.path.join(self.path, "m.npy")
        f_path = os.path.join(self.path, "f.npy")
        temp_c_path = os.path.join(self.path, "tmp_cmd.txt")
        temp_m_path = os.path.join(self.path, "tmp_m.npy")

        # Remove existing results
        if os.path.isfile(f_path):
            os.remove(f_path)

        # Save input to temp files
        np.save(temp_m_path, m)
        with open(temp_c_path, "w") as file:
            file.write(f"compute_f")

        # Safely create input files (avoid imcomplete files)
        os.replace(temp_m_path, m_path)
        os.replace(temp_c_path, c_path)

        # Wait for results
        tic = time.time()
        while True:
            try:
                f = float(np.load(f_path))
                break
            except (OSError, ValueError, EOFError):
                # No file (OSError) or File not complete (ValueError)
                time.sleep(1)
                if time.time() - tic > self.timeout:
                    raise TimeoutError("Cannot get results.")

        # Update history
        self.last_compute_f = [m, f]

        return f

    def compute_g(self, m: np.ndarray) -> np.ndarray:
        """Compute grad

        Args:
            m (np.ndarray): 1D array velocity field

        Returns:
            np.ndarray: grad
        """
        # Ensure the julia code is ready
        self.check_ready()

        # Seach history
        if self.last_compute_g:
            if np.allclose(m, self.last_compute_g[0], 1e-15, 1e-15):
                return self.last_compute_g[1].copy()

        # Define paths
        c_path = os.path.join(self.path, "cmd.txt")
        m_path = os.path.join(self.path, "m.npy")
        f_path = os.path.join(self.path, "f.npy")
        g_path = os.path.join(self.path, "g.npy")
        temp_c_path = os.path.join(self.path, "tmp_cmd.txt")
        temp_m_path = os.path.join(self.path, "tmp_m.npy")

        # Remove existing results
        if os.path.isfile(g_path):
            os.remove(g_path)
        if os.path.isfile(f_path):
            os.remove(f_path)

        # Save input to temp files
        np.save(temp_m_path, m)
        with open(temp_c_path, "w") as file:
            file.write(f"compute_g")

        # Safely create input files (avoid imcomplete files)
        os.replace(temp_m_path, m_path)
        os.replace(temp_c_path, c_path)

        # Wait for results
        tic = time.time()
        while True:
            try:
                g = np.load(g_path)
                f = float(np.load(f_path))
                break
            except (OSError, ValueError, EOFError):
                # No file (OSError) or File not complete (ValueError)
                time.sleep(1)
                if time.time() - tic > self.timeout * 2:
                    raise TimeoutError("Cannot get gradient.")

        # Update history
        self.last_compute_f = [m, f]
        self.last_compute_g = [m, g]

        return g

    def compute_H(self, m: np.ndarray, v: np.ndarray):
        """Compute Hessian action

        Args:
            m (np.ndarray): 1D array velocity field
            v (np.ndarray): 1D array pertubation field

        Returns:
            np.ndarray: H(m)v
        """
        # Ensure the julia code is ready
        self.check_ready()

        # Define paths
        c_path = os.path.join(self.path, "cmd.txt")
        m_path = os.path.join(self.path, "m.npy")
        v_path = os.path.join(self.path, "v.npy")
        Hv_path = os.path.join(self.path, "Hv.npy")
        temp_c_path = os.path.join(self.path, "tmp_cmd.txt")
        temp_m_path = os.path.join(self.path, "tmp_m.npy")
        temp_v_path = os.path.join(self.path, "tmp_v.npy")

        # Remove existing results
        if os.path.isfile(Hv_path):
            os.remove(Hv_path)

        # Save input to temp files
        np.save(temp_m_path, m)
        np.save(temp_v_path, v)
        with open(temp_c_path, "w") as file:
            file.write(f"compute_H")

        # Safely create input files (avoid imcomplete files)
        os.replace(temp_m_path, m_path)
        os.replace(temp_v_path, v_path)
        os.replace(temp_c_path, c_path)

        # Wait for results
        tic = time.time()
        while True:
            try:
                Hv = np.load(Hv_path)
                break
            except (OSError, ValueError, EOFError):
                # No file (OSError) or File not complete (ValueError)
                time.sleep(1)
                if time.time() - tic > self.timeout * 3:
                    raise TimeoutError("Cannot get hessian.")

        return Hv

    def compute_d(self, m: np.ndarray) -> np.ndarray:
        """Compute data

        Args:
            m (np.ndarray): 1D array velocity field

        Returns:
            np.ndarray: data(src_idx, rcv_idx, t)
        """
        # Ensure the julia code is ready
        self.check_ready()

        # Define paths
        c_path = os.path.join(self.path, "cmd.txt")
        m_path = os.path.join(self.path, "m.npy")
        d_path = os.path.join(self.path, "d.npy")
        temp_c_path = os.path.join(self.path, "tmp_cmd.txt")
        temp_m_path = os.path.join(self.path, "tmp_m.npy")

        # Remove existing results
        if os.path.isfile(d_path):
            os.remove(d_path)

        # Save input to temp files
        np.save(temp_m_path, m)
        with open(temp_c_path, "w") as file:
            file.write(f"compute_d")

        # Safely create input files (avoid imcomplete files)
        os.replace(temp_m_path, m_path)
        os.replace(temp_c_path, c_path)

        # Wait for results
        tic = time.time()
        while True:
            try:
                d = np.load(d_path)
                break
            except (OSError, ValueError, EOFError):
                # No file (OSError) or File not complete (ValueError)
                time.sleep(1)
                if time.time() - tic > self.timeout:
                    raise TimeoutError("Cannot get data.")

        return d

    def close(self):
        c_path = os.path.join(self.path, "cmd.txt")
        temp_c_path = os.path.join(self.path, "tmp_cmd.txt")
        with open(temp_c_path, "w") as file:
            file.write("close")
        os.replace(temp_c_path, c_path)


class WaveModelingFG(WaveModeling):
    """When it computes f, it computes g as well."""

    def compute_f(self, m: np.ndarray) -> float:
        """Compute cost.

        Args:
            m (np.ndarray): 1D array velocity field

        Returns:
            float: cost
        """
        # Compute g and save the results to history
        self.compute_g(m)

        # Get f from history
        return self.last_compute_f[1]
