using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using System.Linq;
using System.Web;

namespace FeedBackPlatformWeb.Models
{
    public class Survey
    {
        [Key]
        public int Id { get; set; }
        public string Name { get; set; }
        [ForeignKey("Id")]
        public Category Category { get; set; }
    }
}