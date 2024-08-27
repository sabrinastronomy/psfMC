from __future__ import division, print_function
import os
from collections import OrderedDict
from warnings import warn
from emcee import EnsembleSampler

from multiprocessing import Pool
import multiprocessing as mp
from .utils import print_progress
from .analysis import check_convergence_autocorr, save_posterior_images
from .analysis.images import default_filetypes
from .database import save_database, load_database
from .models import MultiComponentModel


@staticmethod
def log_posterior(param_values):
    """
    log-probability of the posterior distribution for a given set of
    parameters. Note: because of how emcee calls this, the posterior
    function is static. The MultiComponentModel object must be passed using

    :param param_values: Vector of values for all stochastic parameters
    :param kwargs: MUST include 'model' kwarg
    :return: log-likelihood + joint log-prior probability
    """
    model = mc_model
    model.param_values = param_values

    # Calculate prior, and early out for unsupported prior values
    log_priors = model.log_priors()
    if not np.isfinite(log_priors):
        return -np.inf, dict()

    raw_px = model.raw_model()
    conv_px = model.convolved_model(raw_px)
    resid_px = model.residual(conv_px)
    ivm_px = model.composite_ivm(raw_px)
    ps_sub_px = model.point_source_subtracted()

    # Save this evaluation's images as blobs for accumulation.
    # Note: Accumulation must happen in the emcee.sample() loop, since any
    # individual evaluation of log_posterior may be discarded.
    sample_images = {'raw_model': raw_px,
                     'convolved_model': conv_px,
                     'residual': resid_px,
                     'composite_ivm': ivm_px,
                     'point_source_subtracted': ps_sub_px}

    # This is Normal log-likelihood. Consider letting the user choose Normal
    # or Poisson (or maybe others). However, just writing rather than using
    # a distributions.Normal object makes this function about 20% faster
    # TODO: We get positive log-likelihood sometimes, which I guess means
    # the -log(0.5/pi*ivm) term dominates. Maybe errors overestimated?
    ivm_flat = ivm_px[~model.config.bad_px]
    resid_flat = resid_px[~model.config.bad_px]
    log_likelihood = -0.5 * np.sum(resid_flat ** 2 * ivm_flat
                                   - np.log(0.5 / np.pi * ivm_flat))

    # FIXME: kinda a hack. log-likelihood is NaN sometimes, find out why
    # This will just cause MCMC to reject the sample
    if not np.isfinite(log_likelihood):
        return -np.inf, sample_images

    return log_likelihood + log_priors, sample_images


def model_galaxy_mcmc(model_file, output_name=None,
                      write_fits=default_filetypes,
                      iterations=0, burn=0,
                      chains=None, max_iterations=1,
                      convergence_check=check_convergence_autocorr):
    """
    Model the surface brightness distribution of a galaxy or galaxies using
    multi-component Markov Chain Monte Carlo parameter estimation.

    :param model_file: Filename of the model definition file. This should be
        a series of components from psfMC.ModelComponents, with parameters
        supplied as either fixed values or stochastics from psfMC.distributions
    :param output_name: Base name for output files (no file extension). By
        default, files are written out containing the requested image types
        (write_fits param) and the MCMC trace database. If None, use
        out_<model_filename>
    :param write_fits: List of which fits file types to write. By default, raw
        (unconvolved) model, convolved model, model IVM, residual, and point
        sources subtracted.
    :param iterations: Number of retained MCMC samples
    :param burn: Number of discarded (burn-in) MCMC samples
    :param chains: Number of individual chains (walkers) to run. If None, the
        minimum number recommended by emcee will be used. More is better.
    :param max_iterations: Maximum sampler iterations before convergence is
        enforced. Default is 1, which means sampler halts even if not converged.
    :param convergence_check: Function taking an emcee Sampler as argument, and
        returning True or False based on whether the sampler has converged.
        Default function returns True when the autocorrelation time of all
        stochastic variables is < 10% of the total number of samples. Sampling
        will be repeated (increasing the chain length) until convergence check
        is met or until max_iterations iterations are performed.
    """
    if output_name is None:
        output_name = 'out_' + model_file.replace('.py', '')
    output_name += '_{}'
    mp.set_start_method('fork', force=True)
    global mc_model
    mc_model = MultiComponentModel(components=model_file)

    # If chains is not specified, use the minimum number recommended by emcee
    if chains is None:
        chains = 2 * mc_model.num_params + 2



    # FIXME: can't use threads=n right now because model object can't be pickled -- currently fixing this SB
    with Pool() as pool:
        sampler = EnsembleSampler(nwalkers=chains, dim=mc_model.num_params,
                                  lnpostfn=log_posterior, pool=pool)

        # Open database if it exists, otherwise pass backend to create a new one
        db_name = output_name.format('db') + '.fits'

        # TODO: Resume if database exists
        if not os.path.exists(db_name):
            param_vec = mc_model.init_params_from_priors(chains)

            # Run burn-in and discard
            for step, result in enumerate(
                    sampler.sample(param_vec, iterations=burn)):
                # Set new initial sampler state
                param_vec = result[0]
                # No need to retain images from every step, so clear blobs
                sampler.clear_blobs()
                print_progress(step, burn, 'Burning')

            sampler.reset()

            converged = False
            for sampling_iter in range(max_iterations):
                # Now run real samples and retain
                for step, result in enumerate(
                        sampler.sample(param_vec, iterations=iterations)):
                    mc_model.accumulate_images(result[3])
                    # No need to retain images from every step, so clear blobs
                    sampler.clear_blobs()
                    print_progress(step, iterations, 'Sampling')

                if convergence_check(sampler):
                    converged = True
                    break
                else:
                    warn('Not yet converged after {:d} iterations:'
                         .format((sampling_iter + 1)*iterations))
                    convergence_check(sampler, verbose=1)

            # Collect some metadata about the sampling process. These will be saved
            # in the FITS headers of both the output database and the images
            db_metadata = OrderedDict([
                ('MCITER', sampler.chain.shape[1]),
                ('MCBURN', burn),
                ('MCCHAINS', chains),
                ('MCCONVRG', converged),
                ('MCACCEPT', sampler.acceptance_fraction.mean())
            ])
            database = save_database(sampler, mc_model, db_name,
                                     meta_dict=db_metadata)
        else:
            print('Database already contains sampled chains, skipping sampling')
            database = load_database(db_name)

    # Write model output files
    save_posterior_images(mc_model, database, output_name=output_name,
                          filetypes=write_fits)
