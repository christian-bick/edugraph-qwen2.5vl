variable "project_id" {
  description = "The GCP Project ID to deploy resources into."
  type        = string
}

variable "zone" {
  description = "The GCP zone to deploy resources into."
  type        = string
}

variable "instance_name" {
  description = "The name of the Compute Engine VM instance."
  type        = string
  default     = "qwen-training-vm"
}
