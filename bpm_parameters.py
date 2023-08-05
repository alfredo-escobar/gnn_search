Pepeganga_params = {"base": (None, None, None, None, 'base'),
                    "baseadj": (None, None, None, None, 'base'),
                    "GC": (3, None, 0.5, True, 'softmax'),
                    "CT": (10, 0.4, None, False, 'mean'),
                    "GC/CTadj": (3, None, 0.5, True, 'softmax'),
                    }

PepegangaCLIP_params = {"base": (None, None, None, None, 'base'),
                    "baseadj": (None, None, None, None, 'base'),
                    "GC": (3, 0.3, None, False, 'mean'),
                    "CT": (5, 0.3, None, False, 'mean'),
                    "GCadj": (3, 0.3, None, False, 'mean'),
                    "CTadj": (5, 0.2, None, False, 'mean'),
                    }

PepegangaCLIPTrained_params = {"base_trained": (None, None, None, None, 'base'),
                            "base_trainedadj": (None, None, None, None, 'base'),
                            "GC_trained": (5, None, None, True, 'sim'),
                            "CT_trained": (10, 0.5, None, False, 'mean'),
                            "GC_trainedadj": (5, 0.2, None, False, 'mean'),
                            "CT_trainedadj": (6, None, None, True, 'sim'),
                    }

Pepeganga_RQ_params = {"base": (None, None, None, None, 'base'),
                    "baseadj": (None, None, None, None, 'base'),
                    "GC": (10, 0.4, None, False, 'mean'),
                    "GCadj": (10, None, 0.5, True, 'softmax'),
                    }

PepegangaCLIP_RQ_params = {"base": (None, None, None, None, 'base'),
                    "baseadj": (None, None, None, None, 'base'),
                    "GC": (10, 0.2, None, False, 'mean'),
                    "GCadj": (7, 0.2, None, False, 'mean'),
                    }

PepegangaCLIPTrained_RQ_params = {"base_trained": (None, None, None, None, 'base'),
                                "base_trainedadj": (None, None, None, None, 'base'),
                                "GC_trained": (10, 0.2, None, False, 'mean'),
                                "GC_trainedadj": (10, 0.2, None, False, 'mean'),
                                }


Cartier_params = {"base": (None, None, None, None, 'base'),
                "baseadj": (None, None, None, None, 'base'),
                "GC": (15, None, None, True, 'sim'),
                "CT": (7, 0.6, None, False, 'mean'),
                "GCadj": (3, 0.1, None, False, 'mean'),
                "CTadj": (5, 0.4, None, False, 'mean'),
                }

CartierCLIP_params = {"base": (None, None, None, None, 'base'),
                    "baseadj": (None, None, None, None, 'base'),
                    "GC": (7, 0.7, None, False, 'mean'),
                    "CT": (5, 0.9, None, False, 'mean'),
                    "GCadj": (3, 0.2, None, False, 'mean'),
                    "CTadj": (5, None, 3.0, True, 'softmax'),
                    }

CartierCLIPTrained_params = {"base_trained": (None, None, None, None, 'base'),
                            "base_trainedadj": (None, None, None, None, 'base'),
                            "GC_trained": (3, None, 3.0, True, 'softmax'),
                            "CT_trained": (10, 0.7, None, False, 'mean'),
                            "GC_trainedadj": (3, None, 3.0, True, 'softmax'),
                            "CT_trainedadj": (10, 0.2, None, False, 'mean'),
                            }


IKEA_params = {"base": (None, None, None, None, 'base'),
                "baseadj": (None, None, None, None, 'base'),
                "GC": (5, 0.6, None, False, 'mean'),
                "CT": (7, 0.7, None, False, 'mean'),
                "GCadj": (5, 0.6, None, False, 'mean'),
                "CTadj": (7, 0.7, None, False, 'mean'),
                }

IKEACLIP_params = {"base": (None, None, None, None, 'base'),
                    "baseadj": (None, None, None, None, 'base'),
                    "GC": (7, 0.6, None, False, 'mean'),
                    "CT": (5, 0.8, None, False, 'mean'),
                    "GCadj": (5, 0.5, None, False, 'mean'),
                    "CTadj": (7, 0.7, None, False, 'mean'),
                    }

IKEACLIPTrained_params = {"base_trained": (None, None, None, None, 'base'),
                        "base_trainedadj": (None, None, None, None, 'base'),
                        "GC_trained": (5, 0.9, None, False, 'mean'),
                        "CT_trained": (3, 0.9, None, False, 'mean'),
                        "GC/CT_trainedadj": (5, 0.8, None, False, 'mean'),
                        }


UNIQLO_params = {"base": (None, None, None, None, 'base'),
                "baseadj": (None, None, None, None, 'base'),
                "GC": (7, 0.3, None, False, 'mean'),
                "CT": (5, 0.4, None, False, 'mean'),
                "GCadj": (7, 0.4, None, False, 'mean'),
                "CTadj": (3, 0.9, None, False, 'mean'),
                }

UNIQLOCLIP_params = {"base": (None, None, None, None, 'base'),
                    "baseadj": (None, None, None, None, 'base'),
                    "GC": (7, 0.9, None, False, 'mean'),
                    "CT": (5, 0.9, None, False, 'mean'),
                    "GCadj": (10, 0.9, None, False, 'mean'),
                    "CTadj": (10, 0.4, None, False, 'mean'),
                    }

UNIQLOCLIPTrained_params = {"base_trained": (None, None, None, None, 'base'),
                            "base_trainedadj": (None, None, None, None, 'base'),
                            "GC/CT_trained": (3, 0.7, None, False, 'mean'),
                            "GC_trainedadj": (3, 0.7, None, False, 'mean'),
                            "CT_trainedadj": (10, 0.6, None, False, 'mean'),
                            }


WorldMarket_params = {"base": (None, None, None, None, 'base'),
                    "baseadj": (None, None, None, None, 'base'),
                    "GC": (7, None, None, True, 'sim'),
                    "CT": (3, 0.6, None, False, 'mean'),
                    "GCadj": (7, None, None, True, 'sim'),
                    "CTadj": (3, 0.6, None, False, 'mean'),
                    }

WorldMarketCLIP_params = {"base": (None, None, None, None, 'base'),
                    "baseadj": (None, None, None, None, 'base'),
                    "GC": (10, 0.5, None, False, 'mean'),
                    "CT": (3, 0.4, None, False, 'mean'),
                    "GCadj": (10, 0.5, None, False, 'mean'),
                    "CTadj": (10, 0.6, None, False, 'mean'),
                    }

WorldMarketCLIPTrained_params = {"base_trained": (None, None, None, None, 'base'),
                                "base_trainedadj": (None, None, None, None, 'base'),
                                "GC_trained": (10, 0.2, None, False, 'mean'),
                                "CT_trained": (5, 0.3, None, False, 'mean'),
                                "GC_trainedadj": (10, None, 3.0, True, 'softmax'),
                                "CT_trainedadj": (5, 0.3, None, False, 'mean'),
                                }



Homy_params = {"base": (None, None, None, None, 'base'),
                "baseadj": (None, None, None, None, 'base'),
                "GC": (10, None, 0.5, True, 'softmax'),
                "CT": (3, 0.6, None, False, 'mean'),
                "GCadj": (5, None, None, True, 'sim'),
                "CTadj": (3, 0.6, None, False, 'mean'),
                }

HomyCLIP_params = {"base": (None, None, None, None, 'base'),
                    "baseadj": (None, None, None, None, 'base'),
                    "GC/CT": (7, 0.4, None, False, 'mean'),
                    "GC/CTadj": (7, 0.4, None, False, 'mean'),
                    }

HomyCLIPTrained_params = {"base_trained": (None, None, None, None, 'base'),
                        "base_trainedadj": (None, None, None, None, 'base'),
                        "GC_trained": (7, 0.5, None, False, 'mean'),
                        "CT_trained": (10, 0.4, None, False, 'mean'),
                        "GC/CT_trainedadj": (7, 0.5, None, False, 'mean'),
                        }

Homy_RQ_params = {"base": (None, None, None, None, 'base'),
                    "baseadj": (None, None, None, None, 'base'),
                    "GC": (7, 0.9, None, False, 'mean'),
                    "GCadj": (5, None, 3.0, True, 'softmax'),
                    }

HomyCLIP_RQ_params = {"base": (None, None, None, None, 'base'),
                    "baseadj": (None, None, None, None, 'base'),
                    "GC": (3, None, 0.5, True, 'softmax'),
                    "GCadj": (10, 0.2, None, False, 'mean'),
                    }

HomyCLIPTrained_RQ_params = {"base_trained": (None, None, None, None, 'base'),
                            "base_trainedadj": (None, None, None, None, 'base'),
                            "GC_trained": (3, None, 3.0, True, 'softmax'),
                            "GC_trainedadj": (10, None, 0.5, True, 'softmax'),
                            }