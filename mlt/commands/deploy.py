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
import json
import os
import sys
import time
import uuid
import yaml
from string import Template
from subprocess import Popen, PIPE
from termcolor import colored

from mlt.commands import Command
from mlt.utils import (build_helpers, config_helpers, files,
                       kubernetes_helpers, progress_bar,
                       process_helpers, log_helpers)


class DeployCommand(Command):
    def __init__(self, args):
        super(DeployCommand, self).__init__(args)
        self.config = config_helpers.load_config()
        build_helpers.verify_build(self.args)

    def action(self):
        skip_crd_check = self.args['--skip-crd-check']
        if not skip_crd_check:
            kubernetes_helpers.check_crds(exit_on_failure=True)

        if self.args['--no-push']:
            print("Skipping image push")
        else:
            self._push()

        self._deploy_new_container()

        if self.args["--logs"]:
            self._tail_logs()

    def _push(self):
        last_push_duration = files.fetch_action_arg(
            'push', 'last_push_duration')
        self.container_name = files.fetch_action_arg(
            'build', 'last_container')

        self.started_push_time = time.time()
        # TODO: unify these commands by factoring out docker command
        # based on config
        if 'gceProject' in self.config:
            self._push_gke()
        else:
            self._push_docker()

        progress_bar.duration_progress(
            'Pushing ', last_push_duration,
            lambda: self.push_process.poll() is not None)

        # If the push fails, get the stdout and error message and display them
        # to the user, with the error message in red.
        if self.push_process.poll() != 0:
            push_stdout, push_error = self.push_process.communicate()
            print(push_stdout.decode("utf-8"))
            print(colored(push_error.decode("utf-8"), 'red'))
            sys.exit(1)

        with open('.push.json', 'w') as f:
            f.write(json.dumps({
                "last_remote_container": self.remote_container_name,
                "last_push_duration": time.time() - self.started_push_time
            }))

        print("Pushed to {}".format(self.remote_container_name))

    def _push_gke(self):
        self.remote_container_name = "gcr.io/{}/{}".format(
            self.config['gceProject'], self.container_name)
        self._tag()
        self.push_process = Popen(["gcloud", "docker", "--", "push",
                                   self.remote_container_name],
                                  stdout=PIPE, stderr=PIPE)

    def _push_docker(self):
        self.remote_container_name = "{}/{}".format(
            self.config['registry'], self.container_name)
        self._tag()
        self.push_process = Popen(
            ["docker", "push", self.remote_container_name],
            stdout=PIPE, stderr=PIPE)

    def _tag(self):
        process_helpers.run(
            ["docker", "tag", self.container_name, self.remote_container_name])

    @staticmethod
    def _update_app_run_id(app_run_id):
        with open('.push.json', 'r+') as json_file:
            data = json.load(json_file)
            data['app_run_id'] = app_run_id
            json_file.seek(0)
            json.dump(data, json_file, indent=2)
            json_file.truncate()

    def _deploy_new_container(self):
        """Substitutes image, app, run data into k8s-template selected.
           Can also launch user into interactive shell with --interactive flag
        """
        app_name = self.config['name']
        self.namespace = self.config['namespace']
        remote_container_name = files.fetch_action_arg(
            'push', 'last_remote_container')
        if remote_container_name is None:
            raise ValueError("No image found to deploy with. Run a plain "
                             "`mlt deploy` to fix this. Most common reason "
                             "for this is a --no-push was used before "
                             "any image was available to use.")

        print("Deploying {}".format(remote_container_name))
        kubernetes_helpers.ensure_namespace_exists(self.namespace)

        # do template substitution across everything in `k8s-templates` dir
        # replaces things with $ with the vars from template.substitute
        # also patches deployment if interactive mode is set
        self.interactive_deployment_found = False
        app_run_id = str(uuid.uuid4())
        for path, dirs, filenames in os.walk("k8s-templates"):
            self.file_count = len(filenames)
            for filename in filenames:
                with open(os.path.join(path, filename)) as f:
                    template = Template(f.read())
                out = template.substitute(
                    image=remote_container_name,
                    app=app_name, run=app_run_id,
                    **config_helpers.get_template_parameters(self.config))

                interactive, out = self._check_for_interactive_deployment(
                    out, filename)
                self._apply_template(out, filename)
                if interactive:
                    interactive_podname = self._get_most_recent_podname()

            print("\nInspect created objects by running:\n"
                  "$ kubectl get --namespace={} all\n".format(self.namespace))

        self._update_app_run_id(app_run_id)
        # After everything is deployed we'll make a kubectl exec
        # call into our debug container if interactive mode
        if self.args["--interactive"] and self.interactive_deployment_found:
            self._exec_into_pod(interactive_podname)
        elif not self.interactive_deployment_found and \
                self.args['--interactive']:
            raise ValueError("Unable to find deployment to run interactively. "
                             "Multiple deployment files found and bad "
                             "<kube_spec> argument passed.")

    def _check_for_interactive_deployment(self, data, filename):
        """users can have multiple deployment templates
           If only one file in template dir then we'll interactively deploy
           that one, or if filename matches the <kube_spec> optional param
           otherwise we'll throw an error
        """
        interactive = False
        if self.args['--interactive']:
            if self.file_count == 1 or \
                    self.args["<kube_spec>"] == filename:
                data = self._patch_template_spec(data)
                interactive = True
                # if we never hit this point, throw error at end of deployment
                # process because we wanted an interactive deployment
                # but nothing matched criteria
                self.interactive_deployment_found = True
        return interactive, data

    def _apply_template(self, out, filename):
        """take k8s-template data and create deployment in k8s dir"""
        with open(os.path.join('k8s', filename), 'w') as f:
            f.write(out)
        process_helpers.run(
            ["kubectl", "--namespace", self.namespace,
             "apply", "-R", "-f", "k8s"])

    def _get_most_recent_podname(self):
        """don't know of a better way to do this; grab the pod
           created by the job we just deployed
           this gets the most recent pod by name, so we can exec
           into it once everything is done deploying
        """
        pod = process_helpers.run_popen(
            "kubectl get pods --namespace {} ".format(
                self.namespace) +
            "--sort-by=.status.startTime", shell=True
        ).stdout.read().decode('utf-8').strip().splitlines()
        if pod:
            # we want last pod listed, podname is always first
            return pod[-1].split()[0]
        else:
            raise ValueError(
                "No pods found in namespace: {}".format(
                    self.namespace))

    def _patch_template_spec(self, data):
        """Makes `command` of template yaml `sleep infinity`.
           We will also add a `debug=true` label onto this pod for easy
           discovery later.
           # NOTE: for now we only support basic functionality. Only 1
           container in a deployment for now. If there is > 1 container,
           we'll interactively deploy first one we find.
        """
        data = yaml.load(data)

        # references to locations in `data` that contain template and
        # containers locations. This saves calling recursive function
        # twice; once we find a location we store that and move on
        self.template_location = None
        self.containers_location = None

        self._find_metadata_and_container_spec(data)
        if not self.template_location or not self.containers_location:
            raise ValueError("Unable to find 'templates' or 'containers' in "
                             "spec. Unable to deploy interactively without "
                             "these.")

        self.template_location['metadata'] = {'labels': {'debug': 'true'}}
        self.containers_location[0].update(
            {'command':
             ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]})
        return json.dumps(data)

    def _find_metadata_and_container_spec(self, data):
        """recursively finds `metadata` and `containers` location in
           deployment spec, that way we can apply debug label and update
           container deployment to sleep
        """
        # containers and metadata will always be dicts, 98% sure
        data_is_dict = isinstance(data, dict)
        if data_is_dict:
            if not self.template_location:
                if 'template' in data:
                    self.template_location = data['template']
            if not self.containers_location:
                if 'containers' in data:
                    self.containers_location = data['containers']

        if self.template_location and self.containers_location:
            return

        if data_is_dict:
            for key, val in data.items():
                self._find_metadata_and_container_spec(val)
        elif isinstance(data, list):
            for elem in data:
                self._find_metadata_and_container_spec(elem)

    def _exec_into_pod(self, podname):
        """wait til pod comes up and then exec into it"""
        print("Connecting to pod...")
        tries = 0
        while True:
            pod = process_helpers.run_popen(
                "kubectl get pods --namespace {} {} -o json".format(
                    self.namespace, podname),
                shell=True).stdout.read().decode('utf-8')
            if not pod:
                continue

            # check if pod is in running state
            # gcr stores an auth token which could be returned as part
            # of the pod json data
            pod = json.loads(pod)
            if pod.get('items') or pod.get('status'):
                # if there's more than 1 thing returned, we have
                # `pod['items']['status']` otherwise we will always have
                # `pod['status'], so by the second if below we're safe
                # first item is what we care about (or only item)
                if pod.get('items'):
                    pod = pod['items'][0]
                if pod['status']['phase'] == 'Running':
                    break

            if tries == self.args['--retries']:
                raise ValueError("Pod {} not Running".format(podname))
            tries += 1
            print("Retrying {}/{}".format(
                tries, self.args['--retries']))
            time.sleep(1)

        # Get shell to the specified pod running in the user's namespace
        process_helpers.run_popen(
            ["kubectl", "exec", "-it", podname, "--namespace", self.namespace,
             "/bin/bash"], stdout=None, stderr=None).wait()

    def _tail_logs(self):
        log_helpers.call_logs(self.config, self.args)
