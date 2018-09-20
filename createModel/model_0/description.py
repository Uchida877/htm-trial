# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------


## This file defines parameters for a prediction experiment.

###############################################################################
#                                IMPORTANT!!!
# This params file is dynamically generated by the RunExperimentPermutations
# script. Any changes made manually will be over-written the next time
# RunExperimentPermutations is run!!!
###############################################################################


from nupic.frameworks.opf.exp_description_helpers import importBaseDescription

# the sub-experiment configuration
config ={
  'aggregationInfo' : {'seconds': 0, 'fields': [], 'months': 0, 'days': 0, 'years': 0, 'hours': 0, 'microseconds': 0, 'weeks': 0, 'minutes': 0, 'milliseconds': 0},
  'modelParams' : {'tmParams': {'minThreshold': 12, 'activationThreshold': 16, 'pamLength': 5}, 'sensorParams': {'encoders': {u'sine': {'maxval': 1, 'name': 'sine', 'clipInput': True, 'minval': -1, 'n': 521, 'fieldname': 'sine', 'w': 21, 'type': 'ScalarEncoder'}}}, 'spParams': {'synPermInactiveDec': 0.0052850000000000006}, 'clParams': {'alpha': 0.1}},

}

mod = importBaseDescription('../description.py', config)
locals().update(mod.__dict__)
