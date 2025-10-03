terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  zone    = var.zone
}

# Use a local variable to derive the region from the zone (e.g., "us-central1-a" -> "us-central1")
locals {
  region = join("-", slice(split("-", var.zone), 0, 2))
}

# Data source to automatically find the latest PyTorch GPU image
data "google_compute_image" "pytorch_image" {
  project = "deeplearning-platform-release"
  family  = "pytorch-latest-gpu"
}

# Define the Compute Engine VM instance
resource "google_compute_instance" "qwen_training_vm" {
  name         = var.instance_name
  machine_type = "g2-standard-8"
  zone         = var.zone

  # Boot disk configured to use the latest Deep Learning image
  boot_disk {
    initialize_params {
      image = data.google_compute_image.pytorch_image.self_link
      size  = 100 # GB
      type  = "pd-balanced"
    }
    auto_delete = true
  }

  # Attach the L4 GPU
  guest_accelerator {
    type  = "nvidia-l4" # Corrected short name for the type
    count = 1
  }

  # Configure as a Spot VM for cost savings
  scheduling {
    provisioning_model   = "SPOT"
    on_host_maintenance = "TERMINATE"
    automatic_restart   = false # Must be false for Spot VMs
  }

  # Metadata to install the GPU driver on first boot
  metadata = {
    install-gpu-driver = "True"
    enable-osconfig    = "TRUE"
  }

  # Network interface with parameterized subnetwork
  network_interface {
    network    = "default"
    subnetwork = "projects/${var.project_id}/regions/${local.region}/subnetworks/default"
    access_config {}
  }

  # Service account with broad permissions to avoid issues
  service_account {
    email  = "default"
    scopes = ["cloud-platform"]
  }

  # Other settings from your configuration
  can_ip_forward      = false
  deletion_protection = false
  enable_display      = false
  labels = {
    goog-ec-src = "vm_add-tf"
  }
  shielded_instance_config {
    enable_integrity_monitoring = false
    enable_secure_boot          = false
    enable_vtpm                 = false
  }
}