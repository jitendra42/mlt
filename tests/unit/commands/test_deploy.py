#
# -*- coding: utf-8 -*-
#
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

from __future__ import print_function

import uuid
import pytest
from mock import call, MagicMock

from mlt.commands.deploy import DeployCommand
from test_utils.io import catch_stdout


@pytest.fixture
def sleep(patch):
    return patch('time.sleep')


@pytest.fixture
def fetch_action_arg(patch):
    return patch('files.fetch_action_arg', MagicMock(return_value='output'))


@pytest.fixture
def kube_helpers(patch):
    return patch('kubernetes_helpers')


@pytest.fixture
def json_mock(patch):
    return patch('json')


@pytest.fixture
def open_mock(patch):
    return patch('open')


@pytest.fixture
def popen_mock(patch):
    popen_mock = MagicMock()
    popen_mock.return_value.poll.return_value = 0
    return patch('Popen', popen_mock)


@pytest.fixture
def process_helpers(patch):
    return patch('process_helpers')


@pytest.fixture
def progress_bar(patch):
    progress_mock = MagicMock()
    progress_mock.duration_progress.side_effect = lambda x, y, z: print(
        'Pushing ')
    return patch('progress_bar', progress_mock)


@pytest.fixture
def template(patch):
    return patch('Template')


@pytest.fixture
def verify_build(patch):
    return patch('build_helpers.verify_build')


@pytest.fixture
def verify_init(patch):
    return patch('config_helpers.load_config')


@pytest.fixture
def walk_mock(patch):
    return patch('os.walk', MagicMock(return_value=['foo', 'bar']))


@pytest.fixture
def yaml(patch):
    return patch('yaml.load')


def deploy(no_push, skip_crd_check, interactive, extra_config_args, retries=5):
    deploy = DeployCommand(
        {'deploy': True, '--no-push': no_push,
         '--skip-crd-check': skip_crd_check,
         '--interactive': interactive, '--retries': retries,
         '--logs':False})
    deploy.config = {'name': 'app', 'namespace': 'namespace'}
    deploy.config.update(extra_config_args)

    with catch_stdout() as caught_output:
        deploy.action()
        output = caught_output.getvalue()
    return output


def verify_successful_deploy(output, did_push=True, interactive=False):
    """assert pushing, deploying, then objs created, then pushed"""
    pushing = output.find('Pushing ')
    push_skip = output.find('Skipping image push')
    deploying = output.find('Deploying ')
    inspecting = output.find('Inspect created objects by running:\n')
    pushed = output.find('Pushed to ')
    pod_connect = output.find('Connecting to pod...')

    if did_push:
        assert all(var >= 0 for var in (
            deploying, inspecting, pushing, pushed))
        assert deploying < inspecting, pushing < pushed
    else:
        assert all(var == -1 for var in (pushing, pushed))
        assert all(var >= 0 for var in (deploying, inspecting, push_skip))
        assert push_skip < deploying, deploying < inspecting

    if interactive:
        assert pod_connect > inspecting


def test_deploy_gce(walk_mock, progress_bar, popen_mock, open_mock,
                    template, kube_helpers, process_helpers, verify_build,
                    verify_init, fetch_action_arg, json_mock):
    json_mock.load.return_value = {
        'last_remote_container': 'gcr.io/app_name:container_id',
        'last_push_duration': 0.18889}
    output = deploy(
        no_push=False, skip_crd_check=True,
        interactive=False,
        extra_config_args={'gceProject': 'gcr://projectfoo'})
    verify_successful_deploy(output)


def test_deploy_docker(walk_mock, progress_bar, popen_mock, open_mock,
                       template, kube_helpers, process_helpers, verify_build,
                       verify_init, fetch_action_arg, json_mock):
    json_mock.load.return_value = {
        'last_remote_container': 'gcr.io/app_name:container_id',
        'last_push_duration': 0.18889}
    output = deploy(
        no_push=False, skip_crd_check=True,
        interactive=False,
        extra_config_args={'registry': 'dockerhub'})
    verify_successful_deploy(output)


def test_deploy_without_push(walk_mock, progress_bar, popen_mock, open_mock,
                             template, kube_helpers, process_helpers,
                             verify_build, verify_init, fetch_action_arg, json_mock):
    json_mock.load.return_value = {
        'last_remote_container': 'gcr.io/app_name:container_id',
        'last_push_duration': 0.18889}
    output = deploy(
        no_push=True, skip_crd_check=True,
        interactive=False,
        extra_config_args={'gceProject': 'gcr://projectfoo'})
    verify_successful_deploy(output, did_push=False)


def test_deploy_interactive_one_file(walk_mock, progress_bar, popen_mock,
                                     open_mock, template, kube_helpers,
                                     process_helpers, verify_build,
                                     verify_init, fetch_action_arg, sleep,
                                     yaml, json_mock):
    walk_mock.return_value = ['foo']
    yaml.return_value = {
        'template': {'foo': 'bar'}, 'containers': [{'foo': 'bar'}]}
    json_mock.loads.return_value = {'status': {'phase': 'Running'}}
    output = deploy(
        no_push=False, skip_crd_check=True,
        interactive=True,
        extra_config_args={'registry': 'dockerhub'})
    verify_successful_deploy(output, interactive=True)

    # verify that kubectl commands are specifying namespace
    for call_args in process_helpers.run_popen.call_args_list:
        assert isinstance(call_args, type(call))
        assert isinstance(call_args[0], tuple)
        assert len(call_args[0]) > 0
        command = call_args[0][0]
        if command[0] == "kubectl":
            assert "--namespace" in command


def test_deploy_interactive_two_files(walk_mock, progress_bar, popen_mock,
                                      open_mock, template, kube_helpers,
                                      process_helpers, verify_build,
                                      verify_init, fetch_action_arg, sleep,
                                      yaml, json_mock):
    json_mock.loads.return_value = {'status': {'phase': 'Running'}}
    yaml.return_value = {
        'template': {'foo': 'bar'}, 'containers': [{'foo': 'bar'}]}
    output = deploy(
        no_push=False, skip_crd_check=True,
        interactive=True,
        extra_config_args={'registry': 'dockerhub', '<kube_spec>': 'r'})
    verify_successful_deploy(output, interactive=True)


def test_deploy_interactive_pod_not_run(walk_mock, progress_bar, popen_mock,
                                        open_mock, template, kube_helpers,
                                        process_helpers, verify_build,
                                        verify_init, fetch_action_arg, sleep,
                                        yaml, json_mock):
    json_mock.loads.return_value = {'status': {'phase': 'Error'}}
    yaml.return_value = {
        'template': {'foo': 'bar'}, 'containers': [{'foo': 'bar'}]}
    with pytest.raises(ValueError):
        output = deploy(
            no_push=False, skip_crd_check=True,
            interactive=True,
            extra_config_args={'registry': 'dockerhub', '<kube_spec>': 'r'})


def test_deploy_update_app_run_id(open_mock, json_mock):
    run_id = str(uuid.uuid4())
    json_mock_data = {
        'last_remote_container': 'gcr.io/app_name:container_id',
        'last_push_duration': 0.18889}
    json_mock.load.return_value = json_mock_data

    DeployCommand._update_app_run_id(run_id)

    assert json_mock_data['app_run_id'] == run_id


def test_image_push_error(walk_mock, progress_bar, popen_mock, open_mock,
                    template, kube_helpers, process_helpers, verify_build,
                    verify_init, fetch_action_arg, json_mock):
    json_mock.load.return_value = {
        'last_remote_container': 'gcr.io/app_name:container_id',
        'last_push_duration': 0.18889}

    # setup mock to induce and error during the deploy
    popen_mock.return_value.poll.return_value = 1
    output_str = "normal output..."
    error_str = "error message..."
    build_output = MagicMock()
    build_output.decode.return_value = output_str
    error_output = MagicMock()
    error_output.decode.return_value = error_str
    popen_mock.return_value.communicate.return_value = (build_output,
                                                        error_output)

    deploy_cmd = DeployCommand({'deploy': True,
                                '--skip-crd-check': True,
                                '--no-push': False})
    deploy_cmd.config = {'name': 'app', 'namespace': 'namespace'}
    deploy_cmd.config.update({'gceProject': 'gcr://projectfoo'})

    with catch_stdout() as caught_output:
        with pytest.raises(SystemExit):
            deploy_cmd.action()
        output = caught_output.getvalue()

    # assert that we got the normal output, followed by the error message
    output_location = output.find(output_str)
    error_location = output.find(error_str)
    assert all(var >= 0 for var in (output_location, error_location))
    assert output_location < error_location
