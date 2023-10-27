.PHONY: build

dev:
	poetry run python app/onesynth_server.py

build:
	poetry run pyinstaller app/onesynth_server.py --noconsole -y
	cp dist_temp/main/_internal/libtorchaudio.so dist/onesynth_server/_internal
	cp -r dist_temp/main/.dylibs dist/onesynth_server/.dylibs
	cp -r dist_temp/main/_internal/spinvae dist/onesynth_server/_internal/spinvae
	cp -r dist_temp/main/pretrained_weight dist/onesynth_server/pretrained_weight
