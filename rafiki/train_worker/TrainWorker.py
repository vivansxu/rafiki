import time
import logging
import os
import traceback
import pprint

from rafiki.constants import TrainJobStatus, TrialStatus, BudgetType
from rafiki.model import unserialize_model
from rafiki.db import Database
from rafiki.client import Client

from .tuner import propose_with_tuner, train_tuner, create_tuner

logger = logging.getLogger(__name__)

class InvalidTrainJobException(Exception):
    pass

class InvalidModelException(Exception):
    pass

class InvalidBudgetTypeException(Exception):
    pass

class TrainWorker(object):
    def __init__(self, service_id, db=Database()):
        self._service_id = service_id
        self._db = db
        self._client = self._make_client()

    def start(self):
        logger.info('Starting train worker for service of id {}...' \
            .format(self._service_id))

        # Get info about worker service
        with self._db:
            worker = self._db.get_train_job_worker(self._service_id)
            model_id = worker.model_id
            train_job_id = worker.train_job_id

        while True:
            # If budget reached, stop worker service
            with self._db:
                if_budget_reached = self._if_budget_reached(train_job_id, model_id)
                if if_budget_reached:
                    logger.info('Budget for train job has reached')

                    try:
                        self._client.stop_train_job_worker(self._service_id)
                    except Exception:
                        # Throw a warning - likely that another worker has stopped the service
                        logger.warning('Error while stopping train job worker service:')
                        logger.warning(traceback.format_exc())
                        
                    break
            
            # Otherwise, create a new trial
            self._do_new_trial(train_job_id, model_id)

    def stop(self):
        pass

    def _do_new_trial(self, train_job_id, model_id):
        self._db.connect()
        (train_dataset_uri, test_dataset_uri,
            model_serialized, hyperparameters, trial_id) = \
                self._create_new_trial(train_job_id, model_id)
        self._db.disconnect()

        logger.info('Starting trial of ID {} with hyperparameters:'.format(trial_id))
        logger.info(pprint.pformat(hyperparameters))

        try:
            model_inst = unserialize_model(model_serialized)
            model_inst.init(hyperparameters)

            # Train model
            logger.info('Training model...')
            model_inst.train(train_dataset_uri)

            # Evaluate model
            logger.info('Evaluating model...')
            score = model_inst.evaluate(test_dataset_uri)

            logger.info('Score: {}'.format(score))
            
            parameters = model_inst.dump_parameters()
            model_inst.destroy()

            with self._db:
                trial = self._db.get_trial(trial_id)
                self._db.mark_trial_as_complete(
                    trial,
                    score=score,
                    parameters=parameters
                )

        except Exception:
            logger.error('Error while running trial:')
            logger.error(traceback.format_exc())

            with self._db:
                trial = self._db.get_trial(trial_id)
                self._db.mark_trial_as_errored(trial)

    # Returns whether the service reached its budget
    def _if_budget_reached(self, train_job_id, model_id):
        train_job = self._db.get_train_job(train_job_id)
        budget_type = train_job.budget_type
        budget_amount = train_job.budget_amount

        if budget_type == BudgetType.MODEL_TRIAL_COUNT:
            max_trials = budget_amount 
            completed_trials = self._db.get_completed_trials_of_train_job(train_job_id)
            model_completed_trials = [x for x in completed_trials if x.model_id == model_id]
            return len(model_completed_trials) >= max_trials
        else:
            raise InvalidBudgetTypeException()

    # Generates and creates a new trial in the DB
    def _create_new_trial(self, train_job_id, model_id):
        train_job = self._db.get_train_job(train_job_id)
        if train_job is None:
            raise InvalidTrainJobException('ID: {}'.format(train_job_id))

        model = self._db.get_model(model_id)
        if model is None:
            raise InvalidModelException('ID: {}'.format(model_id))
    
        hyperparameters = self._do_hyperparameter_selection(train_job, model)

        trial = self._db.create_trial(
            model_id=model.id, 
            train_job_id=train_job.id, 
            hyperparameters=hyperparameters
        )
        self._db.commit()

        return (
            train_job.train_dataset_uri,
            train_job.test_dataset_uri,
            model.model_serialized,
            hyperparameters,
            trial.id
        )

    # Returns a set of hyperparameter values
    def _do_hyperparameter_selection(self, train_job, model):
        # Pick hyperparameter values
        tuner = self._get_tuner_for_model(train_job, model)
        hyperparameters = propose_with_tuner(tuner)

        return hyperparameters
        
    # Retrieves/creates a tuner for the model for the associated train job
    def _get_tuner_for_model(self, train_job, model):
        # Instantiate tuner
        model_inst = unserialize_model(model.model_serialized)
        hyperparameters_config = model_inst.get_hyperparameter_config()
        tuner = create_tuner(hyperparameters_config)

        # Train tuner with previous trials' scores
        trials = self._db.get_completed_trials_of_train_job(train_job.id)
        model_trial_history = [(x.hyperparameters, x.score) for x in trials if x.model_id == model.id]
        (hyperparameters_list, scores) = [list(x) for x in zip(*model_trial_history)] \
            if len(model_trial_history) > 0 else ([], [])
        tuner = train_tuner(tuner, hyperparameters_list, scores)

        return tuner

    def _make_client(self):
        admin_host = os.environ['ADMIN_HOST']
        admin_port = os.environ['ADMIN_PORT']
        superadmin_email = os.environ['SUPERADMIN_EMAIL']
        superadmin_password = os.environ['SUPERADMIN_PASSWORD']
        client = Client(admin_host=admin_host, admin_port=admin_port)
        client.login(email=superadmin_email, password=superadmin_password)
        return client