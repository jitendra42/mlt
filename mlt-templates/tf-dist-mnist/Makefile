# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: EPL-2.0
#

.PHONY: dependencies

all: main

main: main.py
	@echo "Should run linting and tests before submitting application"

build:
	docker build --build-arg HTTP_PROXY=${HTTP_PROXY} --build-arg HTTPS_PROXY=${HTTPS_PROXY} --build-arg http_proxy=${http_proxy} --build-arg https_proxy=${https_proxy} -t ${CONTAINER_NAME} .

status:
	@echo "TF Job:"
	-@kubectl get tfjob ${JOB_NAME} --namespace ${NAMESPACE} || echo "The job may have been undeployed."
	@echo ""
	@echo "Pods:"
	@kubectl get pods --namespace ${NAMESPACE} -o wide -a -l tf_job_name=${JOB_NAME}