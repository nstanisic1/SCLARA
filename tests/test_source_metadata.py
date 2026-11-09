"""Source-vocabulary and restricted-membership regression contracts."""
import unittest
from analysis.ancient import population_analysis_status, source_flag_categories
from analysis.representation import restricted_records

class SourceMetadata(unittest.TestCase):
    def test_population_status_and_unknown(self):
        source='A19_Cell2023_TableS1.xlsx';field='used in population genetic analyses'
        for raw in ['FALSE related to lov001 1st degree', 'FALSE related to rtp003 and rtp001 1st degree', False]:
            self.assertEqual(population_analysis_status(raw,source,field),'excluded')
        self.assertEqual(population_analysis_status(True,source,field),'included')
        for raw in [None,'','NA','FALSELY labelled']:
            self.assertEqual(population_analysis_status(raw,source,field),'unknown')
        self.assertEqual(population_analysis_status('FALSE reason','another-source',field),'unknown')

    def test_actual_compound_flags_and_source_scope(self):
        self.assertEqual(source_flag_categories('contMT_5pct;lowcov;lowGpAvg','S29_MOESM6.xlsx'),
                         ['genotype_probability','low_coverage','mitochondrial_contamination'])
        self.assertEqual(source_flag_categories('1d_rel_VK527','S29_MOESM6.xlsx'),['kinship'])
        self.assertEqual(source_flag_categories('merged_CGG105920_CGG105921','S01_tables.xlsx'),['sample_identity'])
        self.assertEqual(source_flag_categories('score_tie;shortest_path_to_root','S01_tables.xlsx'),['call_ambiguity'])
        self.assertEqual(source_flag_categories('contMT_5pct','unrecognized-source'),[])

    def test_restricted_membership_preserves_full_cut_provenance(self):
        original={'analysis_units':[{'unit_id':'U','record_ids':['a','b','c'],'curated_record_ids':['b','c'],
                   'localities':[{'record_ids':['a','b'],'weight':0.5},{'record_ids':['c'],'weight':0.5}]}]}
        first=restricted_records(original,{'a','c'})['analysis_units'][0]
        self.assertEqual(first['record_ids'],['a','c']);self.assertEqual(first['curated_record_ids'],['c'])
        self.assertEqual(first['full_cut_curated_record_ids'],['b','c'])
        again=restricted_records({'analysis_units':[first]},{'a'})['analysis_units'][0]
        self.assertEqual(again['curated_record_ids'],[]);self.assertEqual(again['full_cut_curated_record_ids'],['b','c'])
        self.assertEqual(again['localities'][0]['weight'],1)
        self.assertEqual(original['analysis_units'][0]['curated_record_ids'],['b','c'])

if __name__ == '__main__': unittest.main()
