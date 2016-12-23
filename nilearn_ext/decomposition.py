# *- encoding: utf-8 -*-
# Author: Ben Cipollini, Ami Tsuchida
# License: BSD
"""
"""

import os
import os.path as op

import numpy as np
from nilearn import datasets
from nilearn.image import iter_img
from nilearn.masking import apply_mask

from six import string_types
from sklearn.decomposition import FastICA
from sklearn.externals.joblib import Memory
from scipy import stats

from nibabel_ext import NiftiImageWithTerms
from .image import cast_img, clean_img
from .masking import HemisphereMasker, flip_img_lr, GreyMatterNiftiMasker, get_mask_by_key


def generate_components(images, hemi, term_scores=None,
                        n_components=20, random_state=42,
                        out_dir=None, memory=Memory(cachedir='nilearn_cache')):
    """Images: list
    Can be nibabel images, can be file paths.
    """
    # Create grey matter mask from mni template
    target_img = datasets.load_mni152_template()

    # Reshape & mask images
    print("%s: Reshaping and masking images; may take time." % hemi)
    if hemi == 'wb':
        masker = GreyMatterNiftiMasker(target_affine=target_img.affine,
                                       target_shape=target_img.shape,
                                       memory=memory)

    else:  # R and L maskers
        masker = HemisphereMasker(target_affine=target_img.affine,
                                  target_shape=target_img.shape,
                                  memory=memory,
                                  hemisphere=hemi)
    masker = masker.fit()

    # Images may fail to be transformed, and are of different shapes,
    # so we need to trasnform one-by-one and keep track of failures.
    X = []  # noqa
    xformable_idx = np.ones((len(images),), dtype=bool)
    for ii, im in enumerate(images):
        img = cast_img(im, dtype=np.float32)
        img = clean_img(img)
        try:
            X.append(masker.transform(img))
        except Exception as e:
            print("Failed to mask/reshape image %d/%s: %s" % (
                im.get('collection_id', 0),
                op.basename(im),
                e))
            xformable_idx[ii] = False

    # Now reshape list into 2D matrix
    X = np.vstack(X)  # noqa

    # Run ICA and map components to terms
    print("%s: Running ICA; may take time..." % hemi)
    fast_ica = FastICA(n_components=n_components, random_state=random_state)
    fast_ica = memory.cache(fast_ica.fit)(X.T)
    ica_maps = memory.cache(fast_ica.transform)(X.T).T

    # Tomoki's suggestion to normalize components_
    # X ~ ica_maps * fast_ica.components_
    #   = (ica_maps * f) * (fast_ica.components_ / f)
    #   = new_ica_map * new_components_
    C = fast_ica.components_
    factor = np.sqrt(
        np.multiply(C, C).sum(axis=1, keepdims=True))  # (n_components x 1)
    ica_maps = np.multiply(ica_maps, factor)
    fast_ica.components_ = np.multiply(C, 1.0 / (factor + 1e-12))

    if term_scores is not None:
        terms = term_scores.keys()
        term_matrix = np.asarray(term_scores.values())
        term_matrix[term_matrix < 0] = 0
        term_matrix = term_matrix[:, xformable_idx]  # terms x images
        # Don't use the transform method as it centers the data
        ica_terms = np.dot(term_matrix, fast_ica.components_.T).T

    # 2015/12/26 - sign matters for comparison, so don't do this!
    # 2016/02/01 - sign flipping is ok for R-L comparison, but RL concat
    #              may break this.
    # Pretty up the results
    for idx, ic in enumerate(ica_maps):
        if -ic.min() > ic.max():
            # Flip the map's sign for prettiness
            ica_maps[idx] = -ic
            if term_scores:
                ica_terms[idx] = -ica_terms[idx]

    # Create image from maps, save terms to the image directly
    ica_image = NiftiImageWithTerms.from_image(
        masker.inverse_transform(ica_maps))
    if term_scores:
        ica_image.terms = dict(zip(terms, ica_terms.T))

    # Write to disk
    if out_dir is not None:
        out_path = op.join(out_dir, '%s_ica_components.nii.gz' % hemi)
        if not op.exists(op.dirname(out_path)):
            os.makedirs(op.dirname(out_path))
        ica_image.to_filename(out_path)
    return ica_image


def get_dissimilarity_score(dat_pair, scoring='correlation'):
    """
    Given a pair of numpy vectors of the same length, calculate the
    dissimilarity score based on the specified scoring method.

    'correlation': 1-pearson correlation (default)
    'l1norm': l1 distance
    'l2norm': l2 distance
    User can also specify a function.
    """
    assert len(dat_pair) == 2
    assert dat_pair[0].shape == dat_pair[1].shape
    assert dat_pair[0].ndim == 1

    if not isinstance(scoring, string_types):  # function
        score = scoring(dat_pair[0], dat_pair[1])
    elif scoring == 'l1norm':
        score = np.linalg.norm(dat_pair[0] - dat_pair[1], ord=1)
    elif scoring == 'l2norm':
        score = np.linalg.norm(dat_pair[0] - dat_pair[1], ord=2)
    elif scoring == 'correlation':
        score = 1 - stats.stats.pearsonr(dat_pair[0], dat_pair[1])[0]
    else:
        raise NotImplementedError(scoring)

    return score


def compare_components(images, labels, scoring='correlation', flip=True):
    """
    Given a pair of component images with the same n of components, compare
    and give a matrix of dissimilarity scores using the specified method
    (correlation (default), l1norm, l2norm, or any other user-specified function).

    If flip=True, images will be flipped and the lowest dissimilarity score (i.e.
    more similar side) will be given in the resulting score_mat. It also outputs
    sign_mat that indicate any sign-flipping.
    """
    assert len(images) == 2
    assert len(labels) == 2
    assert images[0].shape == images[1].shape
    n_components = images[0].shape[3]  # values @ 0 and 1 are the same
    labels = [l.upper() for l in labels]  # make input labels case insensitive

    print("Scoring components (by %s)" % str(scoring))
    score_mat = np.zeros((n_components, n_components))
    sign_mat = np.zeros((n_components, n_components), dtype=np.int)

    c1_img, c2_img = images[0], images[1]

    # Use lh mask for R vs L to ensure the same size
    if 'R' in labels and 'L' in labels:
        mask = get_mask_by_key("L")
        # Flip R image
        if labels.index('R') == 0:
            c1_img = flip_img_lr(c1_img)
        else:
            c2_img = flip_img_lr(c2_img)

    elif 'R' in labels or 'L' in labels:
        mask = get_mask_by_key("R") if 'R' in labels else get_mask_by_key("L")

    # Use wb mask for wb vs RL comparison
    else:
        mask = get_mask_by_key("wb")

    # Apply mask to get image data.
    c1_dat, c2_dat = [apply_mask(img, mask) for img in (c1_img, c2_img)]

    for c1i, comp1 in enumerate(c1_dat):
        for c2i, comp2 in enumerate(c2_dat):
            # Choose a scoring system.
            # Score should indicate DISSIMILARITY
            # Component sign is meaningless, so try both unless flip = False,
            # and keep track of comparisons that had better score
            # when flipping the sign
            score = np.inf
            signs = [1, -1] if flip else [1]
            for sign in signs:
                c1d, c2d = comp1, sign * comp2
                sc = get_dissimilarity_score((c1d, c2d), scoring=scoring)
                if sc < score:
                    sign_mat[c1i, c2i] = sign
                score = min(score, sc)
            score_mat[c1i, c2i] = score

    return score_mat, sign_mat


def compare_RL(wb_img, scoring="correlation"):
    """Compare R and L side of the whole-brain image using the specified method"""
    n_components = wb_img.shape[3]

    # Use only lh_masker to ensure the same size
    mask = get_mask_by_key("L")
    masked_r = apply_mask(flip_img_lr(wb_img), mask)
    masked_l = apply_mask(wb_img, mask)
    print("Comparing R and L spatial similarity using %s" % scoring)
    score_arr = np.zeros(n_components)
    for i in range(n_components):
        # convert dissimilarity score to similarity score
        score_arr[i] = 1 - get_dissimilarity_score((masked_r[i], masked_l[i]), scoring=scoring)

    return score_arr
