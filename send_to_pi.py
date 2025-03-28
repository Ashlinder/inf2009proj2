from fabric import Connection

def send_model_to_pi(pi_host, pi_password, local_model_path="model_quantized.onnx", remote_model_path="/home/model_quantized.onnx"):
    """
    Sends the quantized ONNX model to a Raspberry Pi using SSH.

    Args:
        pi_host (str): IP address or hostname of the Raspberry Pi.
        pi_password (str): Password for the Raspberry Pi's SSH connection.
        local_model_path (str, optional): Path to the local quantized ONNX model. Defaults to "model_quantized.onnx".
        remote_model_path (str, optional): Destination path on the Raspberry Pi. Defaults to "/home/model_quantized.onnx".
    """
    try:
        conn = Connection(host=pi_host, connect_kwargs={"password": pi_password})
        conn.put(local_model_path, remote=remote_model_path)
        print("Quantized ONNX model sent to Raspberry Pi.")
    except Exception as e:
        print(f"Error: {e}")

# Example usage:
send_model_to_pi("172.20.10.6", "Ashlin0411")