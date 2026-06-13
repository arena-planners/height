cmake_minimum_required(VERSION 3.12)
project(arena_planners_height)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_venv REQUIRED)
find_package(ament_cmake_venv_uv REQUIRED)

# uv selects the interpreter from pyproject's requires-python (>=3.8,<3.9); python3.8 must be
# installed in the image (see FORK_MANIFEST.md). uv_venv takes no interpreter argument.
uv_venv(PROJECTFILE pyproject.toml)

ament_package()
