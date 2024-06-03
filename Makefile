set_env:
	pyenv virtualenv 3.10.6 x_ray_env
	pyenv local x_ray_env

reinstall_package:
	@pip uninstall -y xray || :
	@pip install -e .
