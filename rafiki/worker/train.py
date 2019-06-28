import time
import logging
import os
import traceback
import pickle
import pprint

from rafiki.config import SUPERADMIN_EMAIL, SUPERADMIN_PASSWORD
from rafiki.constants import TrainJobStatus, TrialStatus, BudgetType
from rafiki.model import load_model_class, serialize_knob_config, logger as model_logger
from rafiki.db import Database
from rafiki.client import Client

logger = logging.getLogger(__name__)

class InvalidTrainJobException(Exception): pass
class InvalidModelException(Exception): pass
class InvalidBudgetTypeException(Exception): pass
class InvalidWorkerException(Exception): pass

class TrainWorker(object):
    def __init__(self, service_id, worker_id, db=None):
        if db is None: 
            db = Database()
            
        self._service_id = service_id
        self._db = db
        self._worker_id = worker_id
        self._trial_id = None
        self._sub_train_job_id = None
        self._client = Client(admin_host=os.environ['ADMIN_HOST'], 
                        admin_port=os.environ['ADMIN_PORT'], 
                        advisor_host=os.environ['ADVISOR_HOST'],
                            advisor_port=os.environ['ADVISOR_PORT'])
        self._params_root_dir = os.path.join(os.environ['WORKDIR_PATH'], os.environ['PARAMS_DIR_PATH'])

    def start(self):
        logger.info('Starting train worker for service of ID "{}"...' \
            .format(self._service_id))
            
        # TODO: Break up crazily long & unreadable method
        advisor_id = None
        while True:
            with self._db:
                (self._sub_train_job_id, budget, model_id, model_file_bytes, model_class, \
                    train_job_id, train_dataset_uri, test_dataset_uri) = self._read_worker_info()

                self._get_client().send_event('train_job_worker_started', sub_train_job_id=self._sub_train_job_id)

                if self._if_budget_reached(budget):
                    # If budget reached
                    logger.info('Budget for train job has reached')
                    self._stop_sub_train_job()
                    if advisor_id is not None:
                        self._delete_advisor(advisor_id)
                    break

                # Create a new trial
                logger.info('Creating new trial in DB...')
                trial = self._db.create_trial(
                    sub_train_job_id=self._sub_train_job_id,
                    model_id=model_id,
                    worker_id=self._worker_id
                )
                self._db.commit()
                self._trial_id = trial.id
                logger.info('Created trial of ID "{}" in DB'.format(self._trial_id))
                
            # Don't keep DB connection while training model

            # Perform trial & record results
            score = 0
            try:
                logger.info('Starting trial...')

                # Load model class from bytes
                logger.info('Loading model class...')
                clazz = load_model_class(model_file_bytes, model_class)

                # If not created, create a Rafiki advisor for train worker to propose knobs in trials
                if advisor_id is None:
                    logger.info('Creating Rafiki advisor...')
                    advisor_id = self._create_advisor(clazz)
                    logger.info('Created advisor of ID "{}"'.format(advisor_id))

                # Generate knobs for trial
                logger.info('Requesting for knobs proposal from advisor...')
                knobs = self._get_proposal_from_advisor(advisor_id)
                logger.info('Received proposal of knobs from advisor:')
                logger.info(pprint.pformat(knobs))

                # Mark trial as running in DB
                logger.info('Training & evaluating model...')
                with self._db:
                    trial = self._db.get_trial(self._trial_id)
                    self._db.mark_trial_as_running(trial, knobs)

                def handle_log(log_line, log_lvl):
                    with self._db:
                        trial = self._db.get_trial(self._trial_id)
                        self._db.add_trial_log(trial, log_line, log_lvl)

                (score, params_file_path) = self._train_and_evaluate_model(clazz, knobs, train_dataset_uri, 
                                                                    test_dataset_uri, handle_log)
                logger.info('Trial score: {}'.format(score))
                
                with self._db:
                    logger.info('Marking trial as complete in DB...')
                    trial = self._db.get_trial(self._trial_id)
                    self._db.mark_trial_as_complete(trial, score, params_file_path)

                # Report results of trial to advisor
                try:
                    logger.info('Sending result of trials\' knobs to advisor...')
                    self._feedback_to_advisor(advisor_id, knobs, score)
                except Exception:
                    logger.error('Error while sending result of proposal to advisor:')
                    logger.error(traceback.format_exc())

                self._trial_id = None

            except Exception:
                logger.error('Error while running trial:')
                logger.error(traceback.format_exc())
                logger.info('Marking trial as errored in DB...')

                with self._db:
                    trial = self._db.get_trial(self._trial_id)
                    self._db.mark_trial_as_errored(trial)

                self._trial_id = None
                break # Exit worker upon trial error
            
    def stop(self):
        # If worker is currently running a trial, mark it has terminated
        logger.info('Marking trial as terminated in DB...')
        try:
            if self._trial_id is not None: 
                with self._db:
                    trial = self._db.get_trial(self._trial_id)
                    self._db.mark_trial_as_terminated(trial)

        except Exception:
            logger.error('Error marking trial as terminated:')
            logger.error(traceback.format_exc())

        if self._sub_train_job_id is not None:
            self._get_client().send_event('train_job_worker_stopped', sub_train_job_id=self._sub_train_job_id)

    def _train_and_evaluate_model(self, clazz, knobs, train_dataset_uri, \
                                test_dataset_uri, handle_log):

        # Initialize model
        model_inst = clazz(**knobs)

        # Add logs handlers for trial, including adding handler to root logger 
        # to handle logs emitted during model training with level above INFO
        log_handler = ModelLoggerHandler(handle_log)
        root_logger = logging.getLogger()
        root_logger.addHandler(log_handler)
        py_model_logger = logging.getLogger('{}.trial'.format(__name__))
        py_model_logger.setLevel(logging.INFO)
        py_model_logger.propagate = False # Avoid duplicate logs in root logger
        py_model_logger.addHandler(log_handler)
        model_logger.set_logger(py_model_logger)

        # Train model
        model_inst.train(train_dataset_uri)

        # Evaluate model
        score = model_inst.evaluate(test_dataset_uri)

        # Remove log handlers from loggers for this trial
        root_logger.removeHandler(log_handler)
        py_model_logger.removeHandler(log_handler)

        # Dump and pickle model parameters
        parameters = model_inst.dump_parameters()
        parameters = pickle.dumps(parameters)
        params_file_path = os.path.join(self._params_root_dir, '{}.model'.format(self._trial_id))
        with open(params_file_path, 'wb') as f:
            f.write(parameters)

        model_inst.destroy()

        return (score, params_file_path)

    # Gets proposal of a set of knob values from advisor
    def _get_proposal_from_advisor(self, advisor_id):
        res = self._get_client()._generate_proposal(advisor_id)
        knobs = res['knobs']
        return knobs

    # Feedback result of knobs to advisor
    def _feedback_to_advisor(self, advisor_id, knobs, score):
        self._get_client()._feedback_to_advisor(advisor_id, knobs, score)

    def _stop_sub_train_job(self):
        logger.warn('Stopping sub train job...')
        try:
            self._get_client().send_event('sub_train_job_budget_reached', sub_train_job_id=self._sub_train_job_id)
        except Exception:
            # Throw just a warning - likely that another worker has stopped it
            logger.warn('Error while stopping sub train job:')
            logger.warn(traceback.format_exc())
        
    def _create_advisor(self, clazz):
        # Retrieve knob config for model of worker 
        knob_config = clazz.get_knob_config()
        knob_config_str = serialize_knob_config(knob_config)

        # Create advisor associated with worker
        res = self._get_client()._create_advisor(knob_config_str, advisor_id=self._service_id)
        advisor_id = res['id']
        return advisor_id

    # Delete advisor
    def _delete_advisor(self, advisor_id):
        try:
            self._get_client()._delete_advisor(advisor_id)
        except Exception:
            # Throw just a warning - not critical for advisor to be deleted
            logger.warning('Error while deleting advisor:')
            logger.warning(traceback.format_exc())

    # Returns whether the worker reached its budget (only consider COMPLETED or ERRORED trials)
    def _if_budget_reached(self, budget):
        # By default, budget is model trial count of 5
        max_trials = budget.get(BudgetType.MODEL_TRIAL_COUNT, 5)
        trials = self._db.get_trials_of_sub_train_job(self._sub_train_job_id)
        trials = [x for x in trials if x.status in [TrialStatus.COMPLETED, TrialStatus.ERRORED]]
        return len(trials) >= max_trials

    def _read_worker_info(self):
        worker = self._db.get_train_job_worker(self._service_id)

        if worker is None:
            raise InvalidWorkerException()

        sub_train_job = self._db.get_sub_train_job(worker.sub_train_job_id)
        train_job = self._db.get_train_job(sub_train_job.train_job_id)
        model = self._db.get_model(sub_train_job.model_id)

        if model is None:
            raise InvalidModelException()

        if train_job is None or sub_train_job is None:
            raise InvalidTrainJobException()

        return (
            sub_train_job.id,
            train_job.budget,
            model.id,
            model.model_file_bytes,
            model.model_class,
            train_job.id,
            train_job.train_dataset_uri,
            train_job.test_dataset_uri
        )

    def _get_client(self):
        self._client.login(email=SUPERADMIN_EMAIL, password=SUPERADMIN_PASSWORD)
        return self._client

class ModelLoggerHandler(logging.Handler):
    def __init__(self, handle_log):
        logging.Handler.__init__(self)
        self._handle_log = handle_log

    def emit(self, record):
        log_line = record.msg
        log_lvl = record.levelname
        self._handle_log(log_line, log_lvl)
      