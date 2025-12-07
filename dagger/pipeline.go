package main

import (
	"dagger/itu-sdse-project/internal/dagger"
)

// Base returns a base container with Python and system dependencies installed
func (m *ItuSdseProject) Base() *dagger.Container {
	return dag.Container().
		From("python:3.10-slim").
		WithExec([]string{"apt-get", "update"}).
		WithExec([]string{"apt-get", "install", "-y", "git", "libgomp1"})
}

// WithDeps installs Python dependencies with pip caching for faster re-runs
func (m *ItuSdseProject) WithDeps(source *dagger.Directory) *dagger.Container {
	// Step 1: Start with base container
	ctr := m.Base()

	// Step 2: Mount pip cache for faster re-runs
	ctr = ctr.WithMountedCache("/root/.cache/pip", dag.CacheVolume("pip-cache"))

	// Step 3: Copy ONLY requirements-dagger.txt first (for layer caching)
	ctr = ctr.
		WithFile("/app/requirements-dagger.txt", source.File("requirements-dagger.txt")).
		WithWorkdir("/app").
		WithExec([]string{"pip", "install", "-r", "requirements-dagger.txt"})

	// Step 4: Copy the rest of the code
	ctr = ctr.WithDirectory("/app", source)

	// Step 5: Set PYTHONPATH for module imports
	ctr = ctr.WithEnvVariable("PYTHONPATH", "/app")

	return ctr
}

// Lock installs dependencies and generates a requirements.lock file
func (m *ItuSdseProject) Lock(
	// +ignore=["venv", ".git", ".dvc", "mlruns", "notebooks", "models", "__pycache__", ".pytest_cache", "dagger", "internal"]
	source *dagger.Directory,
) *dagger.File {
	return m.WithDeps(source).
		WithExec([]string{"sh", "-c", "pip freeze > requirements.lock"}).
		File("requirements.lock")
}

// CleanData runs data cleaning (outlier removal, missing value handling)
func (m *ItuSdseProject) CleanData(
	// +ignore=["venv", ".git", ".dvc", "mlruns", "notebooks", "models", "__pycache__", ".pytest_cache", "dagger", "internal"]
	source *dagger.Directory,
) *dagger.Directory {
	return m.WithDeps(source).
		WithExec([]string{"python", "-m", "lead_conversion_prediction.dataset"}).
		Directory("/app")
}

// PrepareData runs feature engineering (imputation, scaling, encoding)
func (m *ItuSdseProject) PrepareData(
	// +ignore=["venv", ".git", ".dvc", "mlruns", "notebooks", "models", "__pycache__", ".pytest_cache", "dagger", "internal"]
	source *dagger.Directory,
) *dagger.Directory {
	return m.WithDeps(source).
		WithExec([]string{"python", "-m", "lead_conversion_prediction.features"}).
		Directory("/app")
}

// Train runs the model training pipeline
func (m *ItuSdseProject) Train(
	// +ignore=["venv", ".git", ".dvc", "mlruns", "notebooks", "models", "__pycache__", ".pytest_cache", "dagger", "internal"]
	source *dagger.Directory,
) *dagger.Directory {
	return m.WithDeps(source).
		WithExec([]string{"python", "-m", "lead_conversion_prediction.modeling.train"}).
		Directory("models")
}

// Package creates a tarball of the trained models
func (m *ItuSdseProject) Package(models *dagger.Directory) *dagger.File {
	return dag.Container().
		From("alpine:latest").
		WithDirectory("/models", models).
		WithWorkdir("/").
		WithExec([]string{"tar", "-cvf", "models.tar", "/models"}).
		File("models.tar")
}

// Upload simulates uploading the package (returns the file for export)
func (m *ItuSdseProject) Upload(packageFile *dagger.File) *dagger.File {
	// In a real scenario, this would upload to S3/GCS/Registry.
	// For now, we just pass it through, allowing the caller to export it.
	return packageFile
}

// Predict runs the model inference (validation)
func (m *ItuSdseProject) Predict(
	// +ignore=["venv", ".git", ".dvc", "mlruns", "notebooks", "__pycache__", ".pytest_cache", "dagger", "internal"]
	source *dagger.Directory,
) *dagger.Container {
	return m.WithDeps(source).
		WithExec([]string{"python", "-m", "lead_conversion_prediction.modeling.predict"})
}

// Pipeline runs the full workflow: CleanData -> PrepareData -> Train -> Package -> Upload
func (m *ItuSdseProject) Pipeline(
	// +ignore=["venv", ".git", ".dvc", "mlruns", "notebooks", "models", "__pycache__", ".pytest_cache", "dagger", "internal"]
	source *dagger.Directory,
) *dagger.File {
	cleaned := m.CleanData(source)
	prepared := m.PrepareData(cleaned)
	models := m.Train(prepared)
	pkg := m.Package(models)
	return m.Upload(pkg)
}
